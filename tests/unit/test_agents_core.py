from __future__ import annotations

from pathlib import Path

import pytest

from fa.agents.claims import check_text, numbers_in
from fa.agents.schemas import AnalystView
from fa.config import AppConfig, Budget, Price
from fa.orchestration.budget import BudgetExceeded, CostTracker, Usage, cost_usd
from fa.orchestration.decision import decide, forecast_signal
from fa.orchestration.scratchpad import Scratchpad, read_events
from fa.tools.registry import NoArgs, RunContext, Tool, ToolError, ToolRegistry

RES = {
    "T1": {
        "result_id": "T1",
        "price": {"last_close": 92.41, "change_20d": 0.1496},
        "inventories": [{"value": 426398, "dev_pct": -0.0742}],
        "verdict": "No measurable edge at 5d (best: ensemble, skill +1.2%)",
    }
}


@pytest.mark.parametrize(
    "text",
    [
        "WTI settled at 92.41 [T1], up 15.0% over 20 days [T1].",
        "Stocks of 426,398 kb are 7.4% below the 5-year average [T1].",
        "Stocks are about 426k barrels [T1].",
        "Stocks are 7% below average [T1].",  # rounding to the digits written
        "The best model's skill was 1.2% [T1].",  # numbers quoted inside strings
        "P10 and P90 over 20d using the 3-2-1 crack on 2026-09-18 and the 200-day average.",
    ],
)
def test_supported_claims_pass(text: str) -> None:
    assert check_text(text, RES) == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Price is 95.10 [T1].", "not found"),
        ("Price is 92.41.", "without citing"),
        ("RSI of 71 is overbought [T9].", "unknown result"),
        ("Stocks are 7.5% below average [T1].", "not found"),
    ],
)
def test_unsupported_claims_fail(text: str, problem: str) -> None:
    (v,) = check_text(text, RES)
    assert problem in v.problem


def test_number_extraction_skips_identifiers() -> None:
    assert [n.raw for n in numbers_in("M1-M2 at T3 with CLZ26 and 5d P50 in 2026")] == []
    assert [n.raw for n in numbers_in("down -2.3% to $91.50, or 1.2bn")] == [
        "-2.3%",
        "$91.50",
        "1.2bn",
    ]


def test_budget_gates(cfg: AppConfig) -> None:
    price = Price(input=2.0, output=10.0, cache_write=2.5, cache_read=0.2)
    assert cost_usd(price, Usage(1_000_000, 100_000)) == pytest.approx(3.0)
    budget = Budget(
        max_usd_per_run=0.05,
        max_usd_backtest=1,
        max_tokens_per_run=10**6,
        preflight_output_fraction=0.5,
    )
    costs = CostTracker({"m": price}, budget)
    costs.charge("a", "m", Usage(10_000, 1_000))  # $0.03
    with pytest.raises(BudgetExceeded):
        costs.preflight("m", 5_000, 4_000)  # $0.01 + $0.02 on top
    with pytest.raises(BudgetExceeded):
        costs.charge("b", "m", Usage(10_000, 1_000))
    assert costs.by_agent == pytest.approx({"a": 0.03, "b": 0.03})


def test_registry_logs_every_result_with_ids(cfg: AppConfig, tmp_path: Path) -> None:
    pad = Scratchpad(tmp_path)
    ctx = RunContext(cfg=cfg, svc=None, symbol="X", as_of=None, scratchpad=pad)  # type: ignore[arg-type]
    reg = ToolRegistry()
    reg.register(Tool("ok", "d", NoArgs, lambda c, a: {"value": 1.23456789}))
    reg.register(Tool("bad", "d", NoArgs, lambda c, a: (_ for _ in ()).throw(ToolError("no data"))))
    r1 = reg.call(ctx, "agent", "ok", {})
    r2 = reg.call(ctx, "agent", "ok", {"unexpected": 1})
    r3 = reg.call(ctx, "agent", "bad", {})
    r4 = reg.call(ctx, "agent", "nope", {})
    assert [r.result_id for r in (r1, r2, r3, r4)] == ["T1", "T2", "T3", "T4"]
    assert r1.payload == {"result_id": "T1", "value": 1.23457} and not r1.is_error
    assert r2.is_error and "invalid arguments" in r2.payload["error"]
    assert r3.payload["error"] == "no data" and r4.is_error
    events = [e for e in read_events(pad.path) if e["event"] == "tool_call"]
    assert [e["result_id"] for e in events] == ["T1", "T2", "T3", "T4"]
    assert reg.schemas(["ok"])[0]["input_schema"]["additionalProperties"] is False


def _view(stance: str, conf: float) -> AnalystView:
    return AnalystView(
        stance=stance, confidence=conf, summary="", key_points=[], red_flags=[], data_gaps=[]
    )  # type: ignore[arg-type]


def test_decision_is_computed_in_code(cfg: AppConfig) -> None:
    views = {
        "supply_demand_analyst": _view("bullish", 1.0),
        "technical_analyst": _view("bullish", 1.0),
        "sentiment_analyst": _view("neutral", 0.5),
        "macro_analyst": _view("bearish", 0.2),
    }
    d = decide(views, None, cfg.agents)
    w = cfg.agents.decision_weights
    expected = w["supply_demand_analyst"] + w["technical_analyst"] - 0.2 * w["macro_analyst"]
    assert d.score == pytest.approx(expected)
    assert d.rating == "Strong Buy" and not d.forecast_edge  # 0.47 >= strong (0.45)
    views["technical_analyst"] = _view("neutral", 0.5)
    assert decide(views, None, cfg.agents).rating == "Buy"  # 0.27
    fc_part = next(c for c in d.components if c.name == "forecast")
    assert fc_part.contribution == 0 and "unavailable" in fc_part.note
    assert (
        decide({k: _view("bearish", 1.0) for k in views}, None, cfg.agents).rating == "Strong Sell"
    )
    assert decide({}, None, cfg.agents).rating == "Hold"
    assert forecast_signal(None) == (0.0, False, "forecast unavailable")
