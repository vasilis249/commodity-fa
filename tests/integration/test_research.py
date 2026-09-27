"""Research loop (`fa ask`) and eval runner (`fa eval`) on a scripted LLM."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fa import evals
from fa.config import AppConfig
from fa.orchestration.research_loop import ask
from fa.orchestration.scratchpad import read_events
from tests.agent_helpers import FakeLLM, offline_service
from tests.forecast_helpers import small_config

AS_OF = date(2026, 6, 1)


@pytest.fixture
def env(cfg: AppConfig, tmp_path: Path):
    small = small_config(cfg)
    return small, offline_service(small, tmp_path), tmp_path


def _ask(env, llm: FakeLLM, question: str = "What was the WTI settle?"):
    cfg, svc, tmp = env
    return ask(cfg, svc, question, "CL=F", llm, as_of=AS_OF, runs_dir=tmp / "runs")


def test_honest_answer_is_verified_and_logged(env) -> None:
    res = _ask(env, FakeLLM())
    a = res.answer
    assert a is not None and a.answerable and res.verified and not res.violations
    assert a.citations == ["T1"] and a.value is not None
    events = read_events(Path(res.scratchpad))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and "answer" in kinds and "run_end" in kinds
    tool_payload = next(e for e in events if e["event"] == "tool_call")["result"]
    assert a.value == tool_payload["price"]["last_close"]
    assert res.cost_usd > 0 and res.calls >= 2


def test_invented_value_is_sent_back_and_fixed(env) -> None:
    llm = FakeLLM(research_mode="lie_once")
    res = _ask(env, llm)
    assert res.verified and res.answer.value != 12345.67
    history = [p for a, p in llm.calls if a == "research_agent"][-1]["messages"]
    feedback = next(
        m["content"]
        for m in history
        if isinstance(m["content"], str) and "code check" in m["content"]
    )
    assert "12345.67" in feedback  # the code check named the unsupported value


def test_persistently_invented_value_is_never_returned(env) -> None:
    res = _ask(env, FakeLLM(research_mode="lie_always"))
    assert not res.verified and res.violations
    assert res.answer.value is None and "12345.67" not in res.answer.answer


def test_decline(env) -> None:
    res = _ask(env, FakeLLM(research_mode="decline"), "What will WTI settle at in 2030?")
    assert res.answer is not None and not res.answer.answerable and res.answer.value is None


# --- eval runner ----------------------------------------------------------------------


def _questions(cfg, svc) -> list[evals.Question]:
    base = {"symbol": "CL=F", "as_of": AS_OF}
    qs = [
        evals.Question(
            id="p",
            category="price",
            question="WTI settle?",
            expect=evals.Expect(type="number"),
            ref=evals.Ref(kind="field", tool="price_technicals", path="price.last_close"),
            **base,
        ),
        evals.Question(
            id="r",
            category="technicals",
            question="WTI RSI?",
            expect=evals.Expect(type="number"),
            ref=evals.Ref(kind="field", tool="price_technicals", path="technicals.rsi14"),
            **base,
        ),
        evals.Question(
            id="n",
            category="unanswerable",
            question="WTI in 2030?",
            expect=evals.Expect(type="unanswerable"),
            ref=evals.Ref(kind="none"),
            **base,
        ),
        evals.Question(
            id="m",
            category="macro",
            question="10y yield?",
            requires=["FRED_API_KEY"],
            expect=evals.Expect(type="number", value=4.0),
            ref=evals.Ref(kind="field", tool="macro_series", path="x"),
            **base,
        ),
    ]
    evals.freeze(cfg, svc, qs[:2])
    return qs


@pytest.fixture
def no_fred(monkeypatch):
    monkeypatch.setattr(
        "fa.evals.load_secrets",
        lambda: type("S", (), {"status": lambda self: {"FRED_API_KEY": False}})(),
    )


def test_reference_solver_scores_everything(env, no_fred) -> None:
    cfg, svc, _tmp = env
    rep = evals.run_evals(cfg, svc, _questions(cfg, svc), solver="reference")
    assert rep.score == 1.0 and rep.run == 3 and rep.skipped == 1 and not rep.drift


def test_llm_solver_is_scored_in_code(env, no_fred) -> None:
    cfg, svc, tmp = env
    rows = []
    rep = evals.run_evals(
        cfg,
        svc,
        _questions(cfg, svc),
        solver="llm",
        llm=FakeLLM(),
        runs_dir=tmp / "runs",
        progress=rows.append,
    )
    by_id = {r.id: r for r in rep.rows}
    assert by_id["p"].correct and by_id["p"].verified  # the fake reads last_close
    assert not by_id["r"].correct  # ...which is not the RSI
    assert not by_id["n"].correct  # it answered instead of declining
    assert by_id["m"].skipped == "needs FRED_API_KEY"
    assert rep.correct == 1 and rep.run == 3 and rep.by_category["price"] == (1, 1)
    assert rep.verified_rate == 1.0 and rep.cost_usd > 0 and len(rows) == 3
    saved = evals.save(rep, tmp / "results")
    assert saved.exists() and "1/3 correct" in saved.with_suffix(".md").read_text()


def test_cost_warning_and_budget_stop(env, no_fred) -> None:
    cfg, svc, tmp = env
    qs = _questions(cfg, svc)
    b = cfg.models.budget
    pricey = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"budget": b.model_copy(update={"est_usd_per_question": 100.0})}
            )
        }
    )
    llm = FakeLLM()
    with pytest.raises(evals.EvalCostWarning):
        evals.run_evals(pricey, svc, qs, solver="llm", llm=llm)
    assert llm.calls == []
    tiny = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"budget": b.model_copy(update={"max_usd_eval": 0.02})}
            )
        }
    )
    rep = evals.run_evals(
        tiny, svc, qs, solver="llm", llm=FakeLLM(), confirm=True, runs_dir=tmp / "runs"
    )
    assert rep.stopped and any(r.skipped == "budget exhausted" for r in rep.rows)
    assert rep.cost_usd <= 0.02 + 1e-9


def test_cli_ask_and_eval_need_a_key(monkeypatch) -> None:
    from typer.testing import CliRunner

    from fa.cli import app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "fa.config.Secrets.model_config",
        {**__import__("fa.config", fromlist=["Secrets"]).Secrets.model_config, "env_file": None},
    )
    r1 = CliRunner().invoke(app, ["ask", "What was the WTI settle?", "--offline"])
    r2 = CliRunner().invoke(app, ["eval", "--offline"])
    assert r1.exit_code == 2 and r2.exit_code == 2 and "reference" in r2.output


def test_unavailable_reference_is_skipped_not_scored(env, no_fred) -> None:
    cfg, svc, tmp = env
    q = evals.Question(
        id="curve",
        category="curve",
        symbol="CL=F",
        as_of=AS_OF,
        question="Contango?",
        expect=evals.Expect(type="choice", value="backwardation"),
        ref=evals.Ref(kind="field", tool="term_structure", path="metrics.structure"),
    )  # no historical curve offline: the data behind the answer is unavailable
    llm = FakeLLM()
    rep = evals.run_evals(cfg, svc, [q], solver="llm", llm=llm, runs_dir=tmp / "runs")
    assert rep.run == 0 and rep.rows[0].skipped.startswith("data unavailable")
    assert llm.calls == []  # nothing spent on an unanswerable-by-data question
