"""Phase 5 acceptance: SMA and forecast backtests vs buy-and-hold on offline data, and the
anonymized agent backtest on a scripted LLM (cost warning, shared cap, long-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fa.backtest.agent_signal import ANON_ANALYSTS, BacktestCostWarning, agent_targets
from fa.backtest.service import run_strategy
from fa.config import AppConfig
from fa.forecasting.dataset import build_dataset
from fa.reports.backtest_text import render
from tests.agent_helpers import FakeLLM, offline_service
from tests.forecast_helpers import small_config
from tests.unit.test_backtest import _leaks


@pytest.fixture
def setup(cfg: AppConfig, tmp_path: Path):
    small = small_config(cfg)
    bt = small.backtest.model_copy(
        update={
            "results_dir": tmp_path / "bt",
            "sma": small.backtest.sma.model_copy(update={"fast": 20, "slow": 60}),
            "forecast": small.backtest.forecast.model_copy(
                update={"model": "drift", "members": ["naive", "drift"]}
            ),
            "agent": small.backtest.agent.model_copy(
                update={"rebalance_every": 20, "max_decisions": 3}
            ),
        }
    )
    small = small.model_copy(update={"backtest": bt})
    svc = offline_service(small, tmp_path)
    ds = build_dataset(svc, "CL=F", tuple(small.forecasting.horizons))
    return small, svc, ds, tmp_path


def test_sma_backtest_vs_buy_and_hold(setup) -> None:
    _cfg, svc, ds, _tmp = setup
    rep = run_strategy(svc, "CL=F", "sma", ds=ds)
    s, b = rep.strategy_metrics, rep.benchmark_metrics
    assert s.start == b.start and s.end == b.end and s.days == b.days
    assert rep.excess["cagr"] == pytest.approx(s.cagr - b.cagr)
    assert rep.edge_p_value is None or 0 <= rep.edge_p_value <= 1
    assert "No measurable edge" in rep.notes[0] or "above buy-and-hold" in rep.notes[0]
    assert s.costs_paid > 0 and b.trades == 1
    saved = json.loads(Path(rep.results_file).read_text())
    assert saved["strategy"] == "sma" and "Paper backtest" in saved["disclaimer"]
    assert Path(rep.results_file).with_suffix(".csv").exists()
    text = render(rep)
    assert "buy & hold" in text and "Sharpe" in text and "Paper backtest" in text


def test_forecast_backtest_runs(setup) -> None:
    _cfg, svc, ds, _tmp = setup
    rep = run_strategy(svc, "CL=F", "forecast", ds=ds, save=False)
    assert rep.results_file is None
    assert any("walk-forward" in n for n in rep.notes)
    assert rep.strategy_metrics.days == rep.benchmark_metrics.days


def test_agent_backtest_on_anonymized_data(setup) -> None:
    cfg, svc, ds, tmp = setup
    llm = FakeLLM(stance="bearish")
    targets, notes, params = agent_targets(svc, "CL=F", ds, llm=llm, runs_dir=tmp / "runs")
    decided = targets.dropna()
    assert params["decisions"] == len(decided) == 3
    assert set(decided.unique()) <= {0.0}  # bearish views, long-only: flat, never short
    agents = {a for a, _ in llm.calls}
    assert set(ANON_ANALYSTS) <= agents and "validator" not in agents
    assert "news_sentiment_analyst" not in agents and "macro_analyst" not in agents
    # nothing identifying reached the model: scan every request in full (system prompt,
    # messages, tool descriptions and schemas, output format)
    for _, request in llm.calls:
        found = _leaks(json.loads(json.dumps(request, default=str)))
        assert not found, f"request leaks {found}"
    assert set(params["models"].values()) <= set(cfg.models.pricing)
    assert any("training cutoff" in n for n in notes)
    assert params["llm_cost_usd"] > 0 and any("anonymized" in n for n in notes)


def test_agent_backtest_bullish_goes_long(setup) -> None:
    _cfg, svc, ds, tmp = setup
    targets, _, _ = agent_targets(svc, "CL=F", ds, llm=FakeLLM(), runs_dir=tmp / "runs")
    assert (targets.dropna() > 0).all()


def test_agent_backtest_warns_before_expensive_runs(setup) -> None:
    cfg, svc, ds, tmp = setup
    agent = cfg.backtest.agent.model_copy(update={"est_usd_per_decision": 10.0})
    cfg2 = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"agent": agent})})
    svc.cfg = cfg2
    llm = FakeLLM()
    with pytest.raises(BacktestCostWarning):
        agent_targets(svc, "CL=F", ds, llm=llm, runs_dir=tmp / "runs")
    assert llm.calls == []  # nothing was spent


def test_agent_backtest_stops_at_the_shared_cap(setup) -> None:
    cfg, svc, ds, tmp = setup
    budget = cfg.models.budget.model_copy(update={"max_usd_backtest": 0.05})
    svc.cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"budget": budget})})
    _targets, notes, params = agent_targets(
        svc, "CL=F", ds, llm=FakeLLM(), confirm=True, runs_dir=tmp / "runs"
    )
    assert params["decisions"] < 3 and any("stopped after" in n for n in notes)
    assert params["llm_cost_usd"] <= 0.05 + 1e-9


def test_cli_backtest_sma_offline(cfg: AppConfig, tmp_path: Path, monkeypatch) -> None:
    from fa.cli import app

    small = small_config(cfg)
    bt = small.backtest.model_copy(
        update={
            "results_dir": tmp_path / "bt",
            "sma": small.backtest.sma.model_copy(update={"fast": 20, "slow": 60}),
        }
    )
    small = small.model_copy(update={"backtest": bt})
    svc = offline_service(small, tmp_path)
    monkeypatch.setattr("fa.data.service.DataService", lambda *a, **k: svc)  # offline fixtures
    res = CliRunner().invoke(app, ["backtest", "CL=F", "-s", "sma", "--offline"])
    assert res.exit_code == 0, res.output
    assert "buy & hold" in res.output and "Paper backtest" in res.output
    assert list((tmp_path / "bt").glob("CL_F_sma_*.json"))


def test_cli_agent_backtest_requires_api_key(monkeypatch) -> None:
    from fa.cli import app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "fa.config.Secrets.model_config",
        {**__import__("fa.config", fromlist=["Secrets"]).Secrets.model_config, "env_file": None},
    )
    res = CliRunner().invoke(app, ["backtest", "CL=F", "-s", "agent", "--offline"])
    assert res.exit_code == 2 and "ANTHROPIC_API_KEY" in res.output


def test_cli_rejects_unknown_strategy() -> None:
    from fa.cli import app

    res = CliRunner().invoke(app, ["backtest", "CL=F", "-s", "magic"])
    assert res.exit_code == 2
