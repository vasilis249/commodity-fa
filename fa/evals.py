"""Evaluation: questions with verifiable answers, answered by the research agent and
scored automatically.

Each question in `evals/questions.jsonl` carries a frozen expected answer and a
reference (`ref`): the tool and field that hold the answer at the question's decision
date. The reference solver recomputes the answer with the tested analytics, which
(a) validates the question set (it must score 100%) and (b) flags drift when data are
revised. Scoring is code, never an LLM:
- number: |got - expected| <= tol (a percent answer is accepted as value/100);
- choice: the normalized answer word equals the expected option;
- unanswerable: the agent must say it cannot answer (answerable = false).
`verified` records whether the agent's value traced to a cited tool result.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fa.agents.schemas import ResearchAnswer
from fa.config import AppConfig, load_secrets
from fa.data.service import DataService
from fa.orchestration.budget import BudgetExceeded, CostTracker
from fa.orchestration.scratchpad import Scratchpad
from fa.tools import market
from fa.tools.registry import RunContext

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
QUESTIONS = EVALS_DIR / "questions.jsonl"
Solver = Literal["llm", "reference"]


# --- question set ---------------------------------------------------------------------


class FieldRef(BaseModel):
    tool: str
    path: str  # e.g. "price.last_close", "inventories[name=cushing_stocks].dev_pct"


class Ref(BaseModel):
    kind: Literal["field", "compare", "none"]
    tool: str | None = None
    path: str | None = None
    a: FieldRef | None = None  # compare: a <op> b -> yes / no
    op: Literal[">", "<"] | None = None
    b: FieldRef | None = None
    b_value: float | None = None  # compare against a constant instead of a field


class Expect(BaseModel):
    type: Literal["number", "choice", "unanswerable"]
    value: float | str | None = None
    options: list[str] = Field(default_factory=list)
    percent: bool = False  # the field is a fraction; "47.4 %" is accepted for 0.474
    tol: float | None = None


class Question(BaseModel):
    id: str
    category: str
    symbol: str
    as_of: date
    question: str
    expect: Expect
    ref: Ref
    source: str = "reference solver (tested analytics code)"
    requires: list[str] = Field(default_factory=list)  # env keys, e.g. FRED_API_KEY


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [
        Question.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]


def save_questions(questions: list[Question], path: Path = QUESTIONS) -> None:
    lines = [q.model_dump_json(exclude_defaults=True) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- reference solver -----------------------------------------------------------------

_SEG = re.compile(r"^(?P<key>[^\[\]]+)?(?:\[(?P<sel>[^\]]+)\])?$")


def extract(payload: Any, path: str) -> Any:
    """Follow "a.b[name=x].c[0]" through dicts and lists of dicts."""
    cur = payload
    for part in re.split(r"\.(?![^\[]*\])", path):  # dots outside [...] only
        m = _SEG.match(part)
        if not m:
            raise KeyError(f"bad path segment {part!r}")
        if m["key"]:
            cur = cur[m["key"]]
        if m["sel"]:
            sel = m["sel"]
            if "=" in sel:
                k, v = sel.split("=", 1)
                cur = next(x for x in cur if str(x.get(k)) == v or _num_eq(x.get(k), v))
            else:
                cur = cur[int(sel)]
    return cur


def _num_eq(x: Any, v: str) -> bool:
    try:
        return float(x) == float(v)
    except (TypeError, ValueError):
        return False


class _Tools:
    """Tool payloads for one (symbol, date), computed once each."""

    def __init__(self, cfg: AppConfig, svc: DataService, q: Question, tmp: Path):
        self.ctx = RunContext(
            cfg=cfg, svc=svc, symbol=q.symbol, as_of=q.as_of, scratchpad=Scratchpad(tmp)
        )
        self.reg = market.build_registry()
        self.cache: dict[str, Any] = {}

    def get(self, ref: FieldRef) -> Any:
        if ref.tool not in self.cache:
            res = self.reg.call(self.ctx, "reference", ref.tool, {})
            if res.is_error:
                raise LookupError(f"{ref.tool}: {res.payload.get('error')}")
            self.cache[ref.tool] = res.payload
        return extract(self.cache[ref.tool], ref.path)


def reference_answer(cfg: AppConfig, svc: DataService, q: Question) -> ResearchAnswer:
    with tempfile.TemporaryDirectory() as tmp:
        tools = _Tools(cfg, svc, q, Path(tmp))
        r = q.ref
        if r.kind == "none":
            return ResearchAnswer(
                plan=[],
                answerable=False,
                value=None,
                unit=None,
                choice=None,
                answer="Not answerable from the tools.",
                citations=[],
            )
        if r.kind == "field":
            assert r.tool and r.path
            v = tools.get(FieldRef(tool=r.tool, path=r.path))
        else:
            assert r.a and r.op and (r.b or r.b_value is not None)
            a = float(tools.get(r.a))
            b = float(tools.get(r.b)) if r.b else float(r.b_value)  # type: ignore[arg-type]
            v = "yes" if (a > b if r.op == ">" else a < b) else "no"
        if isinstance(v, bool):
            v = "yes" if v else "no"
        num = isinstance(v, int | float)
        return ResearchAnswer(
            plan=[f"reference: {r.kind}"],
            answerable=True,
            value=float(v) if num else None,
            unit=None,
            choice=None if num else str(v),
            answer=f"reference value {v}",
            citations=[],
        )


# --- scoring --------------------------------------------------------------------------


def tolerance(e: Expect) -> float:
    if e.tol is not None:
        return e.tol
    assert isinstance(e.value, int | float)
    floor = 0.0006 if e.percent else 0.006  # rounding to 0.1 pp, or to 2 decimals
    return max(0.001 * abs(float(e.value)), floor)


def _written_decimals(x: float) -> int:
    t = repr(x)
    return len(t.split(".")[1]) if "." in t and "e" not in t else 0


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def score(q: Question, ans: ResearchAnswer | None) -> tuple[bool, str]:
    e = q.expect
    if ans is None:
        return False, "no answer"
    if e.type == "unanswerable":
        ok = not ans.answerable and ans.value is None
        return ok, "correctly declined" if ok else "answered a question the tools cannot answer"
    if not ans.answerable:
        return False, "declined an answerable question"
    if e.type == "number":
        if ans.value is None:
            return False, "no (verified) value"
        exp, tol = float(e.value), tolerance(e)  # type: ignore[arg-type]
        # rounding as written is fine ("17.4" for 17.4194), a different number is not
        rounding = 0.5 * 10.0 ** -_written_decimals(ans.value)
        cands = [(ans.value, rounding)] + ([(ans.value / 100, rounding / 100)] if e.percent else [])
        ok = any(abs(c - exp) <= max(tol, r) for c, r in cands)
        return ok, f"got {ans.value:g}, expected {exp:g} ± {tol:.3g}"
    want = _norm(str(e.value))
    got = _norm(ans.choice or "")
    if not got:  # fall back to a single option named in the text
        named = [
            o for o in e.options if re.search(rf"\b{re.escape(_norm(o))}\b", _norm(ans.answer))
        ]
        got = _norm(named[0]) if len(named) == 1 else ""
    return got == want, f"got {got or 'nothing'!r}, expected {want!r}"


# --- running --------------------------------------------------------------------------


class EvalCostWarning(RuntimeError):
    def __init__(self, estimate: float, budget: float, n: int):
        super().__init__(
            f"estimated ${estimate:.2f} for {n} questions exceeds the ${budget:.2f} eval "
            "budget (models.budget.max_usd_eval); confirm to run"
        )


class EvalRow(BaseModel):
    id: str
    category: str
    symbol: str
    as_of: date
    question: str
    expected: float | str | None
    got_value: float | None = None
    got_choice: str | None = None
    answerable: bool | None = None
    answer: str | None = None
    correct: bool = False
    reason: str = ""
    verified: bool | None = None
    reference: float | str | None = None
    drift: bool = False
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None
    scratchpad: str | None = None
    skipped: str | None = None


class EvalReport(BaseModel):
    solver: str
    model: str | None
    prompt_version: str | None
    started: datetime
    questions: int
    run: int
    skipped: int
    correct: int
    score: float | None
    by_category: dict[str, tuple[int, int]]  # category -> (correct, run)
    verified_rate: float | None  # answered numeric values that traced to a citation
    drift: list[str]  # questions whose reference no longer equals the frozen answer
    cost_usd: float
    stopped: str | None = None
    rows: list[EvalRow]


def _ref_value(ans: ResearchAnswer) -> float | str | None:
    return ans.value if ans.value is not None else ans.choice


def _drifted(q: Question, ref: float | str | None) -> bool:
    e = q.expect
    if e.type == "unanswerable":
        return False
    if e.type == "number" and isinstance(ref, float) and isinstance(e.value, int | float):
        return abs(ref - float(e.value)) > tolerance(e)
    return _norm(str(ref)) != _norm(str(e.value))


def run_evals(
    cfg: AppConfig,
    svc: DataService,
    questions: list[Question],
    solver: Solver = "llm",
    llm: Any = None,
    confirm: bool = False,
    runs_dir: Path | None = None,
    progress: Callable[[EvalRow], None] | None = None,
) -> EvalReport:
    from fa.orchestration.research_loop import ask

    keys = {k for k, ok in load_secrets().status().items() if ok}
    todo = [q for q in questions if set(q.requires) <= keys]
    if solver == "llm":
        b = cfg.models.budget
        est = len(todo) * b.est_usd_per_question
        if est > b.max_usd_eval and not confirm:
            raise EvalCostWarning(est, b.max_usd_eval, len(todo))
        if llm is None:
            from fa.agents.base import AnthropicLLM

            llm = AnthropicLLM()
    tracker = CostTracker(
        cfg.models.pricing,
        cfg.models.budget.model_copy(update={"max_usd_per_run": cfg.models.budget.max_usd_eval}),
    )
    rows: list[EvalRow] = []
    stopped = None
    model = prompt_version = None
    for q in questions:
        row = EvalRow(
            id=q.id,
            category=q.category,
            symbol=q.symbol,
            as_of=q.as_of,
            question=q.question,
            expected=q.expect.value,
        )
        if q not in todo:
            row.skipped = f"needs {', '.join(q.requires)}"
            rows.append(row)
            continue
        if stopped:
            row.skipped = "budget exhausted"
            rows.append(row)
            continue
        try:
            ref = reference_answer(cfg, svc, q)
            row.reference = _ref_value(ref)
            row.drift = _drifted(q, row.reference)
        except (LookupError, KeyError, ValueError, StopIteration) as exc:
            if q.expect.type != "unanswerable":
                # the data behind the answer could not be fetched (e.g. a provider rate
                # limit): not the agent's fault, and asking would only spend money
                row.skipped = f"data unavailable: {exc}"
                rows.append(row)
                continue
            ref = None
        if solver == "reference":
            ans = ref
            row.verified = ref is not None
        else:
            try:
                res = ask(
                    cfg,
                    svc,
                    q.question,
                    q.symbol,
                    llm,
                    as_of=q.as_of,
                    runs_dir=runs_dir,
                    costs=tracker,
                )
            except BudgetExceeded as exc:
                stopped = str(exc)
                row.skipped = "budget exhausted"
                rows.append(row)
                continue
            ans = res.answer
            row.verified = res.verified if ans and ans.value is not None else None
            row.cost_usd, row.seconds, row.scratchpad = res.cost_usd, res.seconds, res.scratchpad
            row.error = row.error or res.error
            model, prompt_version = res.model, res.prompt_version
        if ans is not None:
            row.got_value, row.got_choice = ans.value, ans.choice
            row.answerable, row.answer = ans.answerable, ans.answer
        row.correct, row.reason = score(q, ans)
        rows.append(row)
        if progress:
            progress(row)
    ran = [r for r in rows if not r.skipped]
    cats: dict[str, tuple[int, int]] = {}
    for r in ran:
        c, n = cats.get(r.category, (0, 0))
        cats[r.category] = (c + int(r.correct), n + 1)
    numeric = [r for r in ran if r.verified is not None]
    return EvalReport(
        solver=solver,
        model=model,
        prompt_version=prompt_version,
        started=datetime.now(UTC),
        questions=len(questions),
        run=len(ran),
        skipped=len(rows) - len(ran),
        correct=sum(r.correct for r in ran),
        score=sum(r.correct for r in ran) / len(ran) if ran else None,
        by_category=dict(sorted(cats.items())),
        verified_rate=sum(bool(r.verified) for r in numeric) / len(numeric) if numeric else None,
        drift=[r.id for r in ran if r.drift],
        cost_usd=round(tracker.spent_usd, 4),
        stopped=stopped,
        rows=rows,
    )


def freeze(cfg: AppConfig, svc: DataService, questions: list[Question]) -> list[str]:
    """Write reference answers into the questions (for number/choice). Returns changes."""
    changes = []
    for q in questions:
        if q.expect.type == "unanswerable":
            continue
        try:
            v = _ref_value(reference_answer(cfg, svc, q))
        except (LookupError, KeyError, ValueError, StopIteration) as exc:
            changes.append(f"{q.id}: REFERENCE FAILED ({exc!r})")
            continue
        if isinstance(v, float):
            v = float(f"{v:.6g}")
        if v != q.expect.value:
            changes.append(f"{q.id}: {q.expect.value!r} -> {v!r}")
            q.expect.value = v
    return changes


# --- output ---------------------------------------------------------------------------


def to_markdown(r: EvalReport) -> str:
    pct = f"{r.score:.0%}" if r.score is not None else "n/a"
    lines = [
        f"# Eval: {r.correct}/{r.run} correct ({pct})",
        "",
        f"Solver `{r.solver}`"
        + (f", model `{r.model}`, prompt {r.prompt_version}" if r.model else "")
        + f"; {r.skipped} skipped; cost ${r.cost_usd:.3f}; {r.started:%Y-%m-%d %H:%M} UTC.",
    ]
    if r.verified_rate is not None:
        lines.append(f"Numeric answers traced to a cited tool result: {r.verified_rate:.0%}.")
    if r.drift:
        lines.append(f"Reference drift (data revised since freezing): {', '.join(r.drift)}.")
    if r.stopped:
        lines.append(f"Stopped early: {r.stopped}")
    lines += ["", "| category | correct |", "|---|---|"]
    lines += [f"| {c} | {k}/{n} |" for c, (k, n) in r.by_category.items()]
    lines += ["", "| id | ok | expected | got | note |", "|---|---|---|---|---|"]
    for row in r.rows:
        got = row.skipped or (
            "declined" if row.answerable is False else row.got_choice or f"{row.got_value!r}"
        )
        ok = "-" if row.skipped else ("✅" if row.correct else "❌")
        lines.append(
            f"| {row.id} | {ok} | {row.expected!r} | {got} | {row.reason or row.error or ''} |"
        )
    return "\n".join(lines) + "\n"


def save(r: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{r.started:%Y%m%dT%H%M%SZ}_{r.solver}"
    Path(f"{base}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
    Path(f"{base}.md").write_text(to_markdown(r), encoding="utf-8")
    return Path(f"{base}.json")


def dump_row(row: EvalRow) -> str:
    return json.dumps(row.model_dump(mode="json"))
