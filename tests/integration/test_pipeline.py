"""Phase 4 acceptance on a scripted LLM: schema-valid report, every number traced to a
logged tool call, cost tracked and budget enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

from fa.agents.claims import check_fields
from fa.config import AppConfig
from fa.forecasting.service import forecast_dataset
from fa.orchestration.budget import Usage, cost_usd
from fa.orchestration.pipeline import PipelineAborted, run_report
from fa.orchestration.scratchpad import read_events
from fa.reports.render import save, to_markdown
from fa.reports.schema import Report
from tests.agent_helpers import FakeLLM, offline_service
from tests.forecast_helpers import small_config, synthetic_dataset


@pytest.fixture
def setup(cfg: AppConfig, tmp_path: Path, monkeypatch):
    small = small_config(cfg)
    fc = small.forecasting.model_copy(update={"leaderboard_dir": tmp_path / "lb"})
    small = small.model_copy(update={"forecasting": fc})
    ds = synthetic_dataset(small, n=700)
    monkeypatch.setattr(
        "fa.tools.market.run_forecast",
        lambda svc, symbol, as_of=None: forecast_dataset(ds, small, models=["naive", "drift"]),
    )
    return small, offline_service(small, tmp_path), tmp_path


def _run(setup, llm: FakeLLM, **kw) -> Report:
    cfg, svc, tmp = setup
    return run_report(cfg, svc, "CL=F", llm, runs_dir=tmp / "runs", parallel=False, **kw)


def _results_from_scratchpad(path: str) -> dict:
    return {
        e["result_id"]: e["result"] for e in read_events(Path(path)) if e["event"] == "tool_call"
    }


def test_report_is_valid_and_every_number_traces_to_a_logged_tool_call(setup) -> None:
    report = _run(setup, FakeLLM())
    assert Report.model_validate_json(report.model_dump_json()) == report
    logged = _results_from_scratchpad(report.scratchpad)
    assert logged  # tools were really called
    # every LLM-written field (key_numbers are filled by code, not the model)
    texts = report.model_dump(
        include={
            "thesis", "sections", "bull_case", "bear_case",
            "bull_summary", "bear_summary", "risks", "risk_summary",
        },
        exclude={"sections": {"__all__": {"key_numbers"}}},
    )  # fmt: skip
    assert check_fields(texts, logged) == []  # checked against the log, not memory
    assert "[T" in report.thesis


def test_rating_is_the_code_decision_and_exposure_is_capped(setup) -> None:
    report = _run(setup, FakeLLM(stance="bearish"))
    assert report.rating == report.decision.rating
    assert report.decision.score < 0
    assert report.max_exposure <= setup[0].risk.max_position_fraction  # reviewer asked for 90%
    assert "capped" in report.exposure_note


def test_cost_matches_the_scratchpad(setup) -> None:
    report = _run(setup, FakeLLM())
    events = read_events(Path(report.scratchpad))
    llm_calls = [e for e in events if e["event"] == "llm_call"]
    assert report.cost.calls == len(llm_calls) > 0
    assert report.cost.usd == pytest.approx(sum(e["cost_usd"] for e in llm_calls), abs=1e-3)
    one = llm_calls[0]
    price = setup[0].models.pricing[one["model"]]
    assert one["cost_usd"] == pytest.approx(cost_usd(price, Usage(**one["usage"])), abs=1e-6)
    assert {"run_start", "decision", "report", "run_end"} <= {e["event"] for e in events}


def test_unsupported_number_is_sent_back_and_fixed(setup) -> None:
    report = _run(setup, FakeLLM(lie_once={"technical_analyst"}))
    assert report.validation.numeric_violations_fixed >= 1
    assert "12345.67" not in to_markdown(report)
    assert report.validation.sentences_removed == []


def test_persistent_unsupported_number_is_removed(setup) -> None:
    report = _run(setup, FakeLLM(lie_always={"macro_analyst"}))
    assert any("12345.67" in s for s in report.validation.sentences_removed)
    body = report.model_dump_json(exclude={"validation"})
    assert "12345.67" not in body  # gone from every text the reader sees


def test_validator_flag_triggers_revision(setup) -> None:
    llm = FakeLLM(flag_once={"supply_demand_analyst"})
    report = _run(setup, llm)
    assert any(i.startswith("supply_demand_analyst") for i in report.validation.validator_issues)
    calls = [a for a, _ in llm.calls if a == "supply_demand_analyst"]
    assert len(calls) >= 3  # tool turn, first answer, revision after the reviewer's flag


def test_refusing_agent_is_skipped_not_fatal(setup) -> None:
    report = _run(setup, FakeLLM(refuse={"sentiment_analyst"}))
    part = next(c for c in report.decision.components if c.name == "sentiment_analyst")
    assert part.contribution == 0 and "failed" in part.note


def test_structured_output_fallback(setup) -> None:
    llm = FakeLLM(reject_format=True)
    report = _run(setup, llm)
    assert report.rating
    assert any("format" not in p.get("output_config", {}) for _, p in llm.calls)


def test_budget_abort_is_clean(setup) -> None:
    cfg, svc, tmp = setup
    tiny = cfg.models.budget.model_copy(update={"max_usd_per_run": 0.02})
    cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"budget": tiny})})
    with pytest.raises(PipelineAborted) as info:
        run_report(cfg, svc, "CL=F", FakeLLM(), runs_dir=tmp / "runs", parallel=False)
    events = read_events(info.value.scratchpad)
    assert events[-1]["event"] == "run_end"
    assert any(e["event"] == "error" and e.get("stage") == "budget" for e in events)
    assert info.value.spent_usd <= 0.02 + 1e-9


def test_saved_files(setup) -> None:
    report = _run(setup, FakeLLM())
    paths = save(report, setup[2] / "reports")
    assert Report.model_validate_json(paths["json"].read_text()) == report
    md = paths["md"].read_text()
    assert "## Summary" in md and "Not investment advice" in md and "computed in code" in md
    assert paths["html"].read_text().startswith("<!doctype html>")


def test_parallel_analysts_match_sequential(setup) -> None:
    cfg, svc, tmp = setup
    a = run_report(cfg, svc, "CL=F", FakeLLM(), runs_dir=tmp / "r1", parallel=True)
    assert a.rating and len(_results_from_scratchpad(a.scratchpad)) >= 10


def test_cli_report_requires_api_key(monkeypatch) -> None:
    from typer.testing import CliRunner

    from fa.cli import app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "fa.config.Secrets.model_config",
        {**__import__("fa.config", fromlist=["Secrets"]).Secrets.model_config, "env_file": None},
    )
    res = CliRunner().invoke(app, ["report", "CL=F", "--offline"])
    assert res.exit_code == 2 and "ANTHROPIC_API_KEY" in res.output
