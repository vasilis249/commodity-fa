"""Answer one research question: plan -> act (tool calls) -> validate (code) -> answer.

The agent may call any tool listed for `research_agent` in config/agents.yaml. After it
answers, code checks that every number in its text, and the headline `value`, appears
in a tool result it cites. Failures go back to the agent (up to `validator_max_loops`);
whatever still fails is removed: an unverified `value` is never returned.
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fa.agents.base import LLM, AgentError, new_session
from fa.agents.claims import check_fields, check_text
from fa.agents.schemas import ResearchAnswer
from fa.config import AppConfig
from fa.data.service import DataService
from fa.orchestration.budget import BudgetExceeded, CostTracker
from fa.orchestration.scratchpad import Scratchpad
from fa.tools import market
from fa.tools.registry import RunContext

AGENT = "research_agent"
PERCENT_UNITS = {"%", "percent", "pct", "percentage"}


class ResearchResult(BaseModel):
    question: str
    symbol: str
    as_of: date | None
    answer: ResearchAnswer | None
    verified: bool  # value (if any) and every number in the text trace to cited results
    violations: list[str]
    error: str | None = None
    cost_usd: float
    tokens: int
    calls: int
    seconds: float
    model: str
    prompt_version: str
    scratchpad: str


def value_text(ans: ResearchAnswer) -> str | None:
    """The headline value as a cited claim, e.g. '0.809118 [T2]' or '80.9% [T2]'."""
    if ans.value is None:
        return None
    pct = (ans.unit or "").strip().lower() in PERCENT_UNITS
    cites = ", ".join(ans.citations) or "none"
    return f"{ans.value!r}{'%' if pct else ''} [{cites}]"


def verify(ans: ResearchAnswer, results: dict[str, Any]) -> list[str]:
    """Violations: unsupported numbers in the text, an unsupported value, bad citations."""
    text_fields = ans.model_dump(mode="json", exclude={"value", "citations", "unit"})
    problems = [str(v) for v in check_fields(text_fields, results)]
    vt = value_text(ans)
    if vt is not None:
        bad = check_text(vt, results)
        if bad or not ans.citations:
            problems.append(f"value {ans.value!r} is not in the cited results {ans.citations}")
    for rid in ans.citations:
        payload = results.get(rid)
        if payload is None:
            problems.append(f"{rid} is not a tool result in this conversation")
        elif isinstance(payload, dict) and "error" in payload and len(payload) <= 2:
            problems.append(f"{rid} is an error result and cannot support an answer")
    return problems


def _feedback(problems: list[str]) -> str:
    return (
        "A code check could not verify parts of your answer against the tool results you "
        "cited. Cite the result that contains each number and copy it exactly, or remove "
        "it. Then return the full corrected JSON.\n" + "\n".join(f"- {p}" for p in problems)
    )


def _strip_unsupported(ans: ResearchAnswer, results: dict[str, Any]) -> ResearchAnswer:
    text = ans.answer
    for v in check_text(text, results):
        text = text.replace(v.sentence, "")
    text = re.sub(r"\s{2,}", " ", text).strip() or "The answer could not be verified."
    return ans.model_copy(update={"value": None, "answer": text})


def ask(
    cfg: AppConfig,
    svc: DataService,
    question: str,
    symbol: str,
    llm: LLM,
    as_of: date | None = None,
    runs_dir: Path | None = None,
    costs: CostTracker | None = None,
) -> ResearchResult:
    """Answer `question` about `symbol` as of `as_of`. Raises BudgetExceeded when the
    (shared) cost tracker runs out; other agent failures are returned as `error`."""
    t0 = time.monotonic()
    scratch = Scratchpad(runs_dir or cfg.config_dir.parent / ".runs")
    ctx = RunContext(cfg=cfg, svc=svc, symbol=symbol, as_of=as_of, scratchpad=scratch)
    tracker = costs or CostTracker(cfg.models.pricing, cfg.models.budget)
    start = (tracker.spent_usd, tracker.tokens, tracker.calls)
    session = new_session(AGENT, ResearchAnswer, cfg, ctx, market.build_registry(), llm, tracker)
    session.max_steps = cfg.agents.research_loop_max_steps
    scratch.log("run_start", kind="question", question=question, symbol=symbol, as_of=as_of)
    inst = cfg.data.instrument(symbol)
    message = (
        f"Instrument: {symbol} ({inst.name if inst else 'ad-hoc ticker'}). "
        f"Decision date: {as_of or 'latest available'}. Question: {question}"
    )
    ans: ResearchAnswer | None = None
    problems: list[str] = []
    error = None
    try:
        out = session.run(message)
        assert isinstance(out, ResearchAnswer)
        ans = out
        for _ in range(cfg.agents.validator_max_loops):
            problems = verify(ans, scratch.results)
            scratch.log("check", kind="research_answer", violations=problems)
            if not problems:
                break
            revised = session.revise(_feedback(problems))
            assert isinstance(revised, ResearchAnswer)
            ans = revised
        problems = verify(ans, scratch.results)
        if problems:
            ans = _strip_unsupported(ans, scratch.results)
            scratch.log("check", kind="stripped", violations=problems)
    except BudgetExceeded:
        scratch.log("error", stage="budget", spent_usd=tracker.spent_usd)
        raise
    except AgentError as exc:
        error = str(exc)
        scratch.log("error", agent=AGENT, error=error)
    finally:
        scratch.log(
            "run_end",
            spent_usd=round(tracker.spent_usd - start[0], 6),
            tokens=tracker.tokens - start[1],
            calls=tracker.calls - start[2],
        )
    if ans is not None:
        scratch.log("answer", answer=ans.model_dump(mode="json"), verified=not problems)
    return ResearchResult(
        question=question,
        symbol=symbol,
        as_of=as_of,
        answer=ans,
        verified=ans is not None and not problems,
        violations=problems,
        error=error,
        cost_usd=round(tracker.spent_usd - start[0], 6),
        tokens=tracker.tokens - start[1],
        calls=tracker.calls - start[2],
        seconds=round(time.monotonic() - t0, 2),
        model=session.role.model,
        prompt_version=session.prompt.version,
        scratchpad=str(scratch.path),
    )
