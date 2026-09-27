"""Phase 6: every dashboard page runs headless (Streamlit AppTest) on offline fixtures,
with a saved agent report (scripted LLM), a backtest, a leaderboard and a run history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fa import api
from fa.config import AppConfig
from fa.forecasting.dataset import build_dataset
from fa.forecasting.service import forecast_dataset
from tests.agent_helpers import FakeLLM, offline_service
from tests.forecast_helpers import small_config

TIMEOUT = 120


@pytest.fixture
def ws(cfg: AppConfig, tmp_path: Path, monkeypatch) -> api.Workspace:
    small = small_config(cfg)
    bt = small.backtest.model_copy(
        update={
            "results_dir": Path("backtests"),
            "sma": small.backtest.sma.model_copy(update={"fast": 20, "slow": 60}),
        }
    )
    fc = small.forecasting.model_copy(update={"leaderboard_dir": Path("leaderboards")})
    small = small.model_copy(
        update={"config_dir": tmp_path / "config", "backtest": bt, "forecasting": fc}
    )
    svc = offline_service(small, tmp_path)
    work = api.Workspace(small, service_factory=lambda offline: svc)

    def fast_forecast(ws_, symbol, models=None, offline=False, refresh=False):
        ds = build_dataset(svc, symbol, tuple(small.forecasting.horizons))
        return forecast_dataset(ds, small, ["naive", "drift"], unit="USD/bbl", futures=True)

    monkeypatch.setattr("fa.api.forecast", fast_forecast)
    monkeypatch.setattr(
        "fa.tools.market.run_forecast",
        lambda svc_, symbol, as_of=None: fast_forecast(work, symbol),
    )
    monkeypatch.setattr("fa.ui.common.get_workspace", lambda: work)
    monkeypatch.setattr("fa.api.key_status", lambda: {"ANTHROPIC_API_KEY": False})
    st.cache_data.clear()
    return work


def _page(name: str) -> AppTest:
    def script(page: str) -> None:
        from fa.ui import common
        from fa.ui.pages import backtest, forecast, leaderboard, market, reports, runs

        modules = {
            "market": market,
            "forecast": forecast,
            "leaderboard": leaderboard,
            "reports": reports,
            "backtest": backtest,
            "runs": runs,
        }
        common.sidebar()
        modules[page].render()

    at = AppTest.from_function(script, default_timeout=TIMEOUT, args=(name,))
    at.session_state["offline"] = True
    return at


def _ok(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _charts(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_market_page(ws) -> None:
    at = _ok(_page("market"))
    assert "CL=F" in at.header[0].value
    assert any(m.label.startswith("Last") for m in at.metric)
    assert _charts(at)[0]["data"][0]["type"] == "candlestick"


def test_forecast_page_shows_skill_with_every_forecast(ws) -> None:
    at = _ok(_page("forecast"))
    assert not _charts(at)  # nothing until asked
    _button(at, "Run forecast").click()
    _ok(at)
    fan = next(c for c in _charts(at) if "forecast bands" in c["layout"]["title"]["text"])
    notes = [a["text"] for a in fan["layout"]["annotations"]]
    assert len(notes) == len(ws.cfg.forecasting.horizons)
    assert all("skill vs random walk" in n and "DM p" in n for n in notes)
    table = at.dataframe[0].value
    assert {"P10", "P50", "P90", "skill vs random walk", "band coverage", "verdict"} <= set(
        table.columns
    )
    assert any("No measurable edge" in i.value for i in at.info)


def test_leaderboard_page(ws) -> None:
    at = _ok(_page("leaderboard"))
    assert any("No evaluations yet" in i.value for i in at.info)
    api.forecast(ws, "CL=F")  # writes a cached evaluation
    at = _ok(_page("leaderboard"))
    assert len(at.dataframe[0].value) == len(ws.cfg.forecasting.horizons)


def test_reports_page_traces_citations(ws) -> None:
    at = _ok(_page("reports"))
    assert any("No saved reports" in i.value for i in at.info)
    rep, paths = api.run_report(ws, "CL=F", offline=True, llm=FakeLLM())
    assert paths["json"].parent == ws.reports_dir
    at = _ok(_page("reports"))
    assert rep.rating in " ".join(m.value for m in at.markdown)
    cite = next(s for s in at.selectbox if s.label == "Result id")
    events = {e["result_id"]: e for e in api.run_events(rep.scratchpad) if e.get("result_id")}
    assert cite.value in events
    assert at.json, "the cited tool result is shown"
    run_btn = _button(at, "Run report")
    assert run_btn.disabled  # no API key


def test_backtest_page(ws) -> None:
    at = _ok(_page("backtest"))
    _button(at, "Run backtest").click()
    _ok(at)
    assert any(
        "measurable edge" in i.value or "above buy" in i.value for i in [*at.info, *at.success]
    )
    assert any(c["data"][0]["name"] == "sma" for c in _charts(at))
    assert api.list_backtests(ws)
    at = _ok(_page("backtest"))  # the saved tab renders the same backtest without clashing


def test_runs_page(ws) -> None:
    at = _ok(_page("runs"))
    assert any("No runs yet" in i.value for i in at.info)
    api.run_report(ws, "CL=F", offline=True, llm=FakeLLM())
    at = _ok(_page("runs"))
    runs = at.dataframe[0].value
    assert list(runs["kind"]) == ["report"] and runs["cost_usd"].iloc[0] > 0
    assert at.json  # event detail


def test_full_app_default_page(ws) -> None:
    at = AppTest.from_file(
        str(Path(__file__).parents[2] / "app" / "main.py"), default_timeout=TIMEOUT
    )
    at.session_state["offline"] = True
    _ok(at)
    assert "CL=F" in at.header[0].value


def test_api_listings(ws) -> None:
    gas = [i.symbol for i in api.search(ws, "natural gas")]
    assert {"NG=F", "TTF=F"} <= set(gas) and "CL=F" not in gas
    assert "CL=F" in [i.symbol for i in api.search(ws, "crude")]
    assert len(api.search(ws, "")) == len(ws.cfg.data.universe)
    rep = api.run_backtest(ws, "CL=F", "sma", offline=True)
    listed = api.list_backtests(ws)[0]
    assert listed.symbol == "CL=F" and listed.cagr == rep.strategy_metrics.cagr
    loaded, curves = api.load_backtest(listed.path)
    assert loaded.model_dump(exclude={"results_file"}) == rep.model_dump(exclude={"results_file"})
    assert curves is not None and len(curves) == rep.strategy_metrics.days + 1
    api.forecast(ws, "CL=F")
    lb = api.list_leaderboards(ws)[0]
    assert lb.symbol == "CL=F" and {"naive", "drift"} <= set(lb.models)
