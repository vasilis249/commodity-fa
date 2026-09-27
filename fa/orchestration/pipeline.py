"""Report pipeline: analysts -> checks -> debate -> risk -> decision (code) -> synthesis.

Guarantees:
- Every sentence with a number cites a tool result that contains that number. The
  code check (fa.agents.claims) runs on every output; failures go back to the agent
  (up to `validator_max_loops` times). Sentences still failing are removed and listed
  in the report.
- An LLM validator reviews meaning (wrong field, reversed sign, overclaiming) for the
  analysts and the final synthesis; flagged agents get one revision per loop.
- The rating is computed in code (fa.orchestration.decision); the synthesizer explains it.
- The run aborts cleanly on the cost/token budget; the scratchpad keeps everything.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fa.agents.base import LLM, PROMPTS_DIR, AgentError, AgentSession, new_session
from fa.agents.claims import CITE, SENTENCE, Violation, check_fields
from fa.agents.schemas import AnalystView, ResearcherCase, RiskView, Synthesis, ValidatorVerdict
from fa.analytics.snapshot import Snapshot
from fa.config import AppConfig
from fa.data.service import DataService
from fa.forecasting.service import ForecastReport
from fa.orchestration.budget import BudgetExceeded, CostTracker
from fa.orchestration.decision import Decision, decide
from fa.orchestration.scratchpad import Scratchpad
from fa.reports.schema import CostSummary, Report, Section, Validation
from fa.tools import market
from fa.tools.registry import RunContext, ToolRegistry

ANALYSTS = (
    "supply_demand_analyst",
    "technical_analyst",
    "sentiment_analyst",
    "macro_analyst",
    "forecast_interpreter",
)
SECTIONS = (
    ("Supply & demand", "supply_demand_analyst", "fundamentals"),
    ("Technicals", "technical_analyst", "technicals"),
    ("News & sentiment", "sentiment_analyst", "sentiment"),
    ("Macro", "macro_analyst", "macro"),
    ("Forecast", "forecast_interpreter", "forecast"),
)


class PipelineAborted(RuntimeError):
    def __init__(self, reason: str, scratchpad: Path, spent_usd: float):
        super().__init__(reason)
        self.scratchpad = scratchpad
        self.spent_usd = spent_usd


@dataclass
class RunState:
    cfg: AppConfig
    ctx: RunContext
    registry: ToolRegistry
    llm: LLM
    costs: CostTracker
    sessions: dict[str, AgentSession] = field(default_factory=dict)
    fixed: int = 0
    removed: list[str] = field(default_factory=list)
    validator_issues: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    prompts_dir: Path = PROMPTS_DIR
    tool_overrides: dict[str, list[str]] = field(default_factory=dict)

    def session(self, agent: str, model: type[BaseModel]) -> AgentSession:
        s = new_session(
            agent,
            model,
            self.cfg,
            self.ctx,
            self.registry,
            self.llm,
            self.costs,
            prompts_dir=self.prompts_dir,
            tools=self.tool_overrides.get(agent),
        )
        self.sessions[agent] = s
        return s

    @property
    def results(self) -> dict[str, Any]:
        return self.ctx.scratchpad.results


def _feedback(violations: list[Violation]) -> str:
    lines = "\n".join(f"- {v}" for v in violations)
    return (
        "A code check could not verify these numbers against the tool results you cited. "
        "For each one: cite the result that contains the number, copy the number exactly, "
        "or remove it and describe the point in words. Then return the full corrected JSON.\n"
        + lines
    )


def enforce_numbers(state: RunState, session: AgentSession, output: BaseModel) -> BaseModel:
    """Code-check numbers; send failures back; strip what still fails."""
    for _ in range(state.cfg.agents.validator_max_loops):
        violations = check_fields(output, state.results)
        state.ctx.scratchpad.log(
            "check", kind="numeric", agent=session.agent, violations=[str(v) for v in violations]
        )
        if not violations:
            return output
        state.fixed += len(violations)
        output = session.revise(_feedback(violations))
    violations = check_fields(output, state.results)
    if not violations:
        return output
    return _strip(state, session.agent, output, [v.sentence for v in violations])


def _strip(state: RunState, agent: str, output: BaseModel, sentences: list[str]) -> BaseModel:
    def clean(v: Any) -> Any:
        if isinstance(v, str):
            for s in sentences:
                v = v.replace(s, "")
            return re.sub(r"\s{2,}", " ", v).strip()
        if isinstance(v, list):
            return [x for x in (clean(i) for i in v) if x not in ("", None)]
        if isinstance(v, dict):
            return {k: clean(i) for k, i in v.items()}
        return v

    state.removed += [f"{agent}: {s}" for s in sentences]
    state.ctx.scratchpad.log("check", kind="stripped", agent=agent, sentences=sentences)
    return type(output).model_validate(clean(output.model_dump(mode="json")))


STANCE_PREFIX = "stance:"


def _statements(agent: str, output: BaseModel) -> list[dict[str, Any]]:
    out = []
    if isinstance(output, AnalystView):  # the stance itself is a claim to validate
        out.append(
            {
                "agent": agent,
                "statement": (
                    f"{STANCE_PREFIX} {output.stance} (confidence {output.confidence:.2f})"
                ),
            }
        )

    def walk(v: Any) -> None:
        if isinstance(v, str):
            for s in (x.strip() for x in SENTENCE.split(v)):
                if CITE.search(s):
                    out.append({"agent": agent, "statement": s})
        elif isinstance(v, list):
            for i in v:
                walk(i)
        elif isinstance(v, dict):
            for i in v.values():
                walk(i)

    walk(output.model_dump(mode="json"))
    return out


def validate_meaning(state: RunState, outputs: dict[str, BaseModel]) -> dict[str, BaseModel]:
    """LLM validator on meaning; flagged agents revise, then numbers are re-checked."""
    if not state.cfg.agents.agents["validator"].enabled:
        return outputs
    for loop in range(state.cfg.agents.validator_max_loops):
        statements = [st for a, o in outputs.items() for st in _statements(a, o)]
        if not statements:
            return outputs
        cited = sorted(
            {
                rid
                for st in statements
                for g in CITE.findall(st["statement"])
                for rid in g.replace(" ", "").split(",")
            }
        )
        payload = {
            "statements": statements,
            "cited_results": {rid: state.results.get(rid) for rid in cited},
        }
        verdict = state.session("validator", ValidatorVerdict).run(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
        assert isinstance(verdict, ValidatorVerdict)
        issues = [i for i in verdict.issues if i.agent in outputs]
        state.validator_issues += [f"{i.agent}: {i.statement} -> {i.problem}" for i in issues]
        if not issues:
            return outputs
        if loop == state.cfg.agents.validator_max_loops - 1:
            state.unresolved += [f"{i.agent}: {i.statement} -> {i.problem}" for i in issues]
            for agent in sorted({i.agent for i in issues}):
                flagged = [i.statement for i in issues if i.agent == agent]
                outputs[agent] = _drop_unsupported(state, agent, outputs[agent], flagged)
            break
        for agent in sorted({i.agent for i in issues}):
            text = "\n".join(f'- "{i.statement}": {i.problem}' for i in issues if i.agent == agent)
            session = state.sessions[agent]
            revised = session.revise(
                "A reviewer found statements the cited data does not support. Fix or remove "
                f"them and return the full corrected JSON:\n{text}"
            )
            outputs[agent] = enforce_numbers(state, session, revised)
    return outputs


def _drop_unsupported(
    state: RunState, agent: str, output: BaseModel, statements: list[str]
) -> BaseModel:
    """Last resort after the fix loops: remove flagged text; an unsupported stance
    is neutralized (confidence 0) so it cannot move the code-computed rating."""
    if isinstance(output, AnalystView) and any(s.startswith(STANCE_PREFIX) for s in statements):
        output = output.model_copy(update={"confidence": 0.0})
        state.ctx.scratchpad.log("check", kind="stance_neutralized", agent=agent)
    text = [s for s in statements if not s.startswith(STANCE_PREFIX)]
    return _strip(state, agent, output, text) if text else output


def _instrument_line(ctx: RunContext, snap: Snapshot) -> str:
    return (
        f"Instrument: {snap.symbol} ({snap.name}), unit {snap.unit}. "
        f"Decision time: {snap.as_of} settlement. Use your tools, then return your view."
    )


def run_analysts(
    state: RunState,
    snap: Snapshot,
    parallel: bool,
    analysts: tuple[str, ...] = ANALYSTS,
    message: str | None = None,
    validate: bool = True,
) -> dict[str, AnalystView]:
    enabled = [a for a in analysts if state.cfg.agents.agents[a].enabled]
    sessions = {a: state.session(a, AnalystView) for a in enabled}
    msg = message or _instrument_line(state.ctx, snap)

    def one(agent: str) -> tuple[str, BaseModel | None]:
        try:
            out = sessions[agent].run(msg)
            return agent, enforce_numbers(state, sessions[agent], out)
        except AgentError as exc:
            state.ctx.scratchpad.log("error", agent=agent, error=str(exc))
            return agent, None

    if parallel:
        with ThreadPoolExecutor(max_workers=len(enabled)) as pool:
            done = list(pool.map(one, enabled))
    else:
        done = [one(a) for a in enabled]
    views: dict[str, BaseModel] = {a: v for a, v in done if v is not None}
    if validate:
        views = validate_meaning(state, views)
    return {a: v for a, v in views.items() if isinstance(v, AnalystView)}


def run_debate(
    state: RunState, views: dict[str, AnalystView], rounds: int
) -> tuple[ResearcherCase | None, ResearcherCase | None]:
    if rounds <= 0 or not views:
        return None, None
    analysts = json.dumps({a: v.model_dump() for a, v in views.items()}, ensure_ascii=False)
    bull_s = state.session("bull_researcher", ResearcherCase)
    bear_s = state.session("bear_researcher", ResearcherCase)
    intro = f"Round 1 of {rounds}. Analyst views:\n{analysts}\nMake your case."
    bull = enforce_numbers(state, bull_s, bull_s.run(intro))
    bear = enforce_numbers(state, bear_s, bear_s.run(intro))
    for r in range(2, rounds + 1):
        new_bull = bull_s.revise(
            f"Round {r} of {rounds}. The bear's latest case:\n{bear.model_dump_json()}\n"
            "Update your case and rebut."
        )
        new_bear = bear_s.revise(
            f"Round {r} of {rounds}. The bull's latest case:\n{bull.model_dump_json()}\n"
            "Update your case and rebut."
        )
        bull = enforce_numbers(state, bull_s, new_bull)
        bear = enforce_numbers(state, bear_s, new_bear)
    assert isinstance(bull, ResearcherCase) and isinstance(bear, ResearcherCase)
    return bull, bear


def run_risk(
    state: RunState,
    views: dict[str, AnalystView],
    bull: ResearcherCase | None,
    bear: ResearcherCase | None,
    limit: float,
) -> tuple[RiskView | None, float, str]:
    if not state.cfg.agents.agents["risk_reviewer"].enabled:
        return None, limit, "risk reviewer disabled: code limit applies"
    s = state.session("risk_reviewer", RiskView)
    msg = json.dumps(
        {
            "analysts": {a: v.model_dump() for a, v in views.items()},
            "bull": bull.model_dump() if bull else None,
            "bear": bear.model_dump() if bear else None,
        },
        ensure_ascii=False,
    )
    try:
        out = enforce_numbers(state, s, s.run(f"Inputs:\n{msg}\nReview the risks."))
    except AgentError as exc:
        state.ctx.scratchpad.log("error", agent="risk_reviewer", error=str(exc))
        return None, limit, "risk reviewer failed: code limit applies"
    assert isinstance(out, RiskView)
    if out.suggested_max_exposure > limit:
        return (
            out,
            limit,
            f"reviewer suggested {out.suggested_max_exposure:.0%}; "
            f"capped at the code limit {limit:.0%}",
        )
    return out, out.suggested_max_exposure, out.exposure_rationale


def _pct(x: float | None, signed: bool = True) -> str | None:
    if x is None:
        return None
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.0f}%"


def _key_numbers(
    snap: Snapshot, key: str, fc: ForecastReport | None
) -> dict[str, float | str | None]:
    """Headline numbers per section, straight from code (formatted for readers)."""
    p, t = snap.price, snap.technicals
    out: dict[str, float | str | None] = {}
    if key == "fundamentals":
        for inv in snap.inventories:
            out[f"{inv.name} vs 5y avg"] = _pct(inv.dev_pct)
        if snap.cot:
            out["managed money net, 3y percentile"] = snap.cot.mm_net_pctile_3y
        if snap.curve:
            out["curve"] = snap.curve.metrics.structure
            out["roll yield (annualized)"] = _pct(snap.curve.metrics.roll_yield_ann)
        if snap.crack:
            out["3-2-1 crack (USD/bbl)"] = snap.crack.value
    elif key == "technicals":
        out = {
            "last settle": p.last_close,
            "20d change": _pct(p.change_20d),
            "RSI14": t.rsi14,
            "trend": t.trend,
            "vol regime": t.vol_regime,
        }
    elif key == "sentiment" and snap.cot:
        out = {"managed money net, 3y percentile": snap.cot.mm_net_pctile_3y}
    elif key == "forecast" and fc:
        out = {f"{h.horizon}d P(up)": _pct(h.p_up, signed=False) for h in fc.horizons}
        out["edge vs random walk"] = "yes" if any(h.skill.edge for h in fc.horizons) else "no"
    return out


LIVE_WINDOW_DAYS = 3


def _lookahead_warning(as_of: date, today: date) -> str | None:
    """The agents' own training data may cover what happened after a past as_of."""
    if (today - as_of).days <= LIVE_WINDOW_DAYS:
        return None
    return (
        f"Past-dated report (as of {as_of}). The LLM agents may already know what happened "
        "after this date from their training data, even though every tool is cut at the "
        "decision time. Treat this report as illustrative, not as evidence of skill; "
        "anonymized backtests are the valid test."
    )


def run_report(
    cfg: AppConfig,
    svc: DataService,
    symbol: str,
    llm: LLM,
    as_of: date | None = None,
    runs_dir: Path | None = None,
    debate_rounds: int | None = None,
    parallel: bool = True,
) -> Report:
    root = cfg.config_dir.parent
    scratch = Scratchpad(runs_dir or root / ".runs")
    ctx = RunContext(cfg=cfg, svc=svc, symbol=symbol, as_of=as_of, scratchpad=scratch)
    costs = CostTracker(cfg.models.pricing, cfg.models.budget)
    state = RunState(cfg, ctx, market.build_registry(), llm, costs)
    rounds = cfg.agents.debate_rounds if debate_rounds is None else debate_rounds
    scratch.log(
        "run_start",
        symbol=symbol,
        as_of=as_of,
        roles=cfg.models.model_dump(include={"roles", "agent_roles", "budget"}),
        debate_rounds=rounds,
    )
    try:
        snap: Snapshot = market._snapshot(ctx)
        try:
            fc: ForecastReport | None = market._forecast(ctx)
        except Exception as exc:  # the report still runs; the forecast is reported missing
            scratch.log("error", stage="forecast", error=str(exc))
            fc = None
        views = run_analysts(state, snap, parallel)
        bull, bear = run_debate(state, views, rounds)
        risk_view, exposure, exposure_note = run_risk(
            state, views, bull, bear, snap.risk.position.fraction
        )
        decision: Decision = decide(views, fc, cfg.agents)
        scratch.log("decision", decision=decision.model_dump(mode="json"))
        synthesis = _synthesize(state, views, bull, bear, risk_view, decision, fc, exposure)
    except BudgetExceeded as exc:
        scratch.log("error", stage="budget", error=str(exc), spent_usd=costs.spent_usd)
        raise PipelineAborted(str(exc), scratch.path, costs.spent_usd) from exc
    finally:
        scratch.log(
            "run_end", spent_usd=round(costs.spent_usd, 6), tokens=costs.tokens, calls=costs.calls
        )

    sections = [
        Section(
            title=title,
            agent=agent,
            view=views.get(agent),
            text=getattr(synthesis, key)
            if synthesis
            else (views[agent].summary if agent in views else "Unavailable."),
            key_numbers=_key_numbers(snap, key, fc),
        )
        for title, agent, key in SECTIONS
    ]
    models = {
        a: f"{s.role.model} ({s.role.effort}), prompt v{s.prompt.version}"
        for a, s in state.sessions.items()
    }
    report = Report(
        symbol=snap.symbol,
        name=snap.name,
        unit=snap.unit,
        as_of=snap.as_of,
        generated_at=datetime.now(UTC),
        run_id=scratch.run_id,
        rating=decision.rating,
        conviction=decision.conviction,
        decision=decision,
        thesis=synthesis.thesis if synthesis else "Synthesis unavailable.",
        forecast=fc.horizons if fc else [],
        forecast_eval=f"walk-forward {fc.eval_start}..{fc.eval_end}, {fc.folds} folds"
        if fc
        else "unavailable",
        sections=sections,
        bull_case=bull,
        bear_case=bear,
        bull_summary=synthesis.bull_case if synthesis else "",
        bear_summary=synthesis.bear_case if synthesis else "",
        risks=risk_view.risks if risk_view else [],
        risk_summary=synthesis.risks if synthesis else "",
        max_exposure=exposure,
        exposure_note=exposure_note,
        validation=Validation(
            numeric_violations_fixed=state.fixed,
            sentences_removed=state.removed,
            validator_issues=state.validator_issues,
            unresolved_validator_issues=state.unresolved,
        ),
        data_sources=sorted(
            {f"{s.provider}:{s.key} ({s.fetched_at:%Y-%m-%d %H:%M} UTC)" for s in snap.sources}
        ),
        data_notes=snap.notes + (fc.notes if fc else []),
        models=models,
        cost=CostSummary(
            usd=round(costs.spent_usd, 4),
            tokens=costs.tokens,
            calls=costs.calls,
            by_agent={k: round(v, 4) for k, v in costs.by_agent.items()},
            budget_usd=cfg.models.budget.max_usd_per_run,
        ),
        scratchpad=str(scratch.path),
        lookahead_warning=_lookahead_warning(snap.as_of, svc.now().date()),
    )
    scratch.log("report", rating=report.rating, conviction=report.conviction)
    return report


def _synthesize(
    state: RunState,
    views: dict[str, AnalystView],
    bull: ResearcherCase | None,
    bear: ResearcherCase | None,
    risk_view: RiskView | None,
    decision: Decision,
    fc: ForecastReport | None,
    exposure: float,
) -> Synthesis | None:
    if not state.cfg.agents.agents["synthesizer"].enabled:
        return None
    payload = {
        "decision_computed_in_code": decision.model_dump(mode="json"),
        "analysts": {a: v.model_dump() for a, v in views.items()},
        "bull": bull.model_dump() if bull else None,
        "bear": bear.model_dump() if bear else None,
        "risk": risk_view.model_dump() if risk_view else None,
        "forecast_verdicts": [h.skill.verdict for h in fc.horizons] if fc else [],
        "max_exposure_fraction": exposure,
    }
    s = state.session("synthesizer", Synthesis)
    try:
        out = enforce_numbers(state, s, s.run(json.dumps(payload, ensure_ascii=False)))
        out = validate_meaning(state, {"synthesizer": out})["synthesizer"]
    except AgentError as exc:
        state.ctx.scratchpad.log("error", agent="synthesizer", error=str(exc))
        return None
    assert isinstance(out, Synthesis)
    return out
