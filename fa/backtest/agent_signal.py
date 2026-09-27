"""LLM-agent backtest signal on anonymized data.

At each decision date t (every `rebalance_every` sessions, the last `max_decisions`):
1. build the point-in-time snapshot as of t, plus the walk-forward forecast made at t;
2. expose them through the anonymized tools (fa.backtest.anonymize);
3. run the analysts (and an optional bull/bear debate) with the anonymized prompts;
4. compute the rating in code, mapped to a target weight.
The forecast only counts if its skill on forecasts already settled by t beats the
random walk (p < alpha), so no future information enters the decision.

Cost control: the estimated cost (decisions x est_usd_per_decision) must fit
`models.budget.max_usd_backtest` unless the caller confirms; a shared CostTracker caps
the whole backtest at that amount regardless.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from fa.agents.base import LLM, PROMPTS_DIR, AnthropicLLM
from fa.analytics.snapshot import build_snapshot
from fa.backtest.anonymize import (
    ANON_AGENT_TOOLS,
    AnonState,
    build_anon_registry,
    forecast_rows_at,
    past_skill,
)
from fa.backtest.signals import forecast_targets
from fa.config import AppConfig
from fa.data.calendars import calendar_for_exchange
from fa.data.rolls import expiries
from fa.data.service import DataService
from fa.forecasting.dataset import Dataset
from fa.orchestration.budget import BudgetExceeded, CostTracker
from fa.orchestration.decision import decide
from fa.orchestration.pipeline import RunState, run_analysts, run_debate
from fa.orchestration.scratchpad import Scratchpad
from fa.tools.registry import RunContext

ANON_PROMPTS = PROMPTS_DIR / "anon"
ANON_ANALYSTS = ("supply_demand_analyst", "technical_analyst", "forecast_interpreter")
RATING_WEIGHT = {"Strong Buy": 1.0, "Buy": 0.5, "Hold": 0.0, "Sell": -0.5, "Strong Sell": -1.0}
ANON_MESSAGE = (
    "Asset: anonymized. Decision time: the close of the latest session. "
    "Use your tools, then return your view."
)


class BacktestCostWarning(RuntimeError):
    def __init__(self, estimate: float, budget: float, decisions: int):
        super().__init__(
            f"estimated ${estimate:.2f} for {decisions} agent decisions exceeds the "
            f"${budget:.2f} backtest budget (models.budget.max_usd_backtest); confirm to run"
        )
        self.estimate, self.budget, self.decisions = estimate, budget, decisions


def decision_positions(origins: list[int], every: int, max_n: int) -> list[int]:
    """Every `every` sessions (on forecast origins), keeping the most recent `max_n`."""
    chosen: list[int] = []
    for p in origins:
        if not chosen or p - chosen[-1] >= every:
            chosen.append(p)
    return chosen[-max_n:]


def _sessions_to_roll(svc: DataService, symbol: str, ds: Dataset, pos: int) -> int | None:
    inst = svc.instrument(symbol)
    if inst.roll_rule.value == "none":
        return None
    d = ds.dates[pos].date()
    exp = expiries(
        inst.roll_rule, d, date(d.year + 1, 12, 31), calendar_for_exchange(inst.exchange)
    )
    nxt = [e for e in exp.index if e.date() >= d]
    return int(pd.bdate_range(d, nxt[0].date()).size - 1) if nxt else None


def rebase_factor(close: pd.Series, pos: int) -> float:
    """100 / the last positive close at or before the first decision (known then).
    Fails closed: absolute price levels must never reach the prompt."""
    known = close.iloc[: pos + 1]
    positive = known[known > 0]
    if positive.empty:
        raise ValueError("no positive close at or before the first decision; cannot rebase")
    return 100.0 / float(positive.iloc[-1])


def cutoff_notes(cfg: AppConfig, decision_dates: list[pd.Timestamp]) -> tuple[list[str], dict]:
    """Which models made the decisions, and how many decisions they may remember."""
    agents = [*ANON_ANALYSTS, "bull_researcher", "bear_researcher"]
    roles = {cfg.models.agent_roles.get(a, cfg.models.default_role) for a in agents}
    specs = {r: cfg.models.roles[r] for r in sorted(roles)}
    models = {r: sp.model for r, sp in specs.items()}
    notes = []
    for model in sorted(set(models.values())):
        cutoffs = [sp.training_cutoff for sp in specs.values() if sp.model == model]
        cut = next((c for c in cutoffs if c is not None), None)
        if cut is None:
            notes.append(
                f"LLM memory: the training cutoff of {model} is not configured "
                "(models.yaml roles.*.training_cutoff); treat every decision as possibly "
                "inside its training data, where anonymization is the only defence."
            )
        else:
            before = sum(d.date() <= cut for d in decision_dates)
            notes.append(
                f"LLM memory: {before} of {len(decision_dates)} decisions fall on or before "
                f"the training cutoff of {model} ({cut}); only anonymization protects those."
            )
    return notes, models


def _forecast_component(
    skill: dict[str, float | None] | None, row: dict[str, float], alpha: float
) -> tuple[float, bool, str]:
    """Counts only with a significant positive edge on forecasts settled before t."""
    if not skill or "p_up" not in row:
        return 0.0, False, "no settled forecast history"
    s, p = skill.get("pinball_skill_vs_random_walk"), skill.get("dm_p_value")
    if s is None or p is None or s <= 0 or p >= alpha:
        return 0.0, False, "no significant edge on settled forecasts"
    return 2 * row["p_up"] - 1, True, f"settled-forecast edge (skill {s:+.1%}, p={p:.3g})"


def agent_targets(
    svc: DataService,
    symbol: str,
    ds: Dataset,
    llm: LLM | None = None,
    confirm: bool = False,
    runs_dir: Path | None = None,
) -> tuple[pd.Series, list[str], dict[str, Any]]:
    cfg: AppConfig = svc.cfg
    a = cfg.backtest.agent
    fs = cfg.backtest.forecast
    _, ev = forecast_targets(ds, cfg, fs, cfg.backtest.long_short)
    origins = sorted(set(ev.positions[fs.horizon].tolist()))
    positions = decision_positions(origins, a.rebalance_every, a.max_decisions)
    budget = cfg.models.budget.max_usd_backtest
    estimate = len(positions) * a.est_usd_per_decision
    if estimate > budget and not confirm:
        raise BacktestCostWarning(estimate, budget, len(positions))

    llm = llm or AnthropicLLM()
    shared = CostTracker(
        cfg.models.pricing, cfg.models.budget.model_copy(update={"max_usd_per_run": budget})
    )
    runs_dir = runs_dir or cfg.config_dir.parent / ".runs"
    factor = rebase_factor(ds.close, positions[0]) if positions else 1.0
    targets = pd.Series(float("nan"), index=ds.dates)
    notes: list[str] = []
    decided = 0
    for i, p in enumerate(positions):
        t = ds.dates[p]
        snap = build_snapshot(svc, symbol, as_of=t.date(), include_curve=False)
        skill = past_skill(ev, fs.model, p, cfg.forecasting)
        state_data = AnonState(
            snapshot=snap,
            factor=factor,
            forecast_rows=forecast_rows_at(ev, fs.model, t),
            past_skill=skill,
            sessions_to_roll=_sessions_to_roll(svc, symbol, ds, p),
        )
        ctx = RunContext(
            cfg=cfg,
            svc=svc,
            symbol="ASSET",
            as_of=t.date(),
            scratchpad=Scratchpad(runs_dir, run_id=f"bt{i:02d}"),
        )
        ctx.cache["anon"] = state_data
        state = RunState(
            cfg,
            ctx,
            build_anon_registry(),
            llm,
            shared,
            prompts_dir=ANON_PROMPTS,
            tool_overrides=ANON_AGENT_TOOLS,
        )
        try:
            views = run_analysts(
                state,
                snap,
                parallel=True,
                analysts=ANON_ANALYSTS,
                message=ANON_MESSAGE,
                validate=False,
            )
            run_debate(state, views, a.debate_rounds)
        except BudgetExceeded as exc:
            notes.append(f"stopped after {decided} decisions: {exc}")
            break
        fsig = _forecast_component(
            skill.get(fs.horizon),
            state_data.forecast_rows.get(fs.horizon, {}),
            cfg.forecasting.skill.significance_alpha,
        )
        decision = decide(views, None, cfg.agents, forecast_component=fsig)
        weight = RATING_WEIGHT[decision.rating]
        if not cfg.backtest.long_short:
            weight = max(weight, 0.0)
        targets.iloc[p] = weight
        ctx.scratchpad.log(
            "backtest_decision", t=str(t.date()), rating=decision.rating, target=weight
        )
        decided += 1
    if positions:
        targets.iloc[: positions[0]] = float("nan")
    memory_notes, models = cutoff_notes(cfg, [ds.dates[p] for p in positions[:decided]])
    notes += [
        f"{decided} anonymized agent decisions every {a.rebalance_every} sessions; "
        f"LLM cost ${shared.spent_usd:.3f} (cap ${budget:.2f}).",
        "Agents saw rebased prices, ratios and percentiles only (no names, dates, units or "
        "sources). Residual risk: the price path's shape can still hint at famous episodes, "
        "and which data exist (e.g. a processing margin) reveals the commodity class.",
        *memory_notes,
    ]
    params = {
        "agent": a.model_dump(),
        "decisions": decided,
        "llm_cost_usd": round(shared.spent_usd, 4),
        "models": models,
    }
    return targets, notes, params
