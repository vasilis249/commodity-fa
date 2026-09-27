"""Backtester: execution timing, costs, metrics, signal causality, anonymized payloads."""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fa.analytics.snapshot import Snapshot
from fa.backtest.agent_signal import (
    _forecast_component,
    cutoff_notes,
    decision_positions,
    rebase_factor,
)
from fa.backtest.anonymize import (
    ANON_AGENT_TOOLS,
    ANON_TOOLS,
    AnonState,
    build_anon_registry,
    past_skill,
)
from fa.backtest.engine import CostModel, buy_and_hold, pnl_bars, run_backtest
from fa.backtest.metrics import edge_pvalue, summarize
from fa.backtest.service import compare
from fa.backtest.signals import forecast_targets, sma_crossover, vol_targeted
from fa.config import AppConfig, SMASettings
from fa.forecasting.leakage import LeakageError, assert_causal
from fa.orchestration.scratchpad import Scratchpad
from fa.tools.registry import RunContext
from tests.conftest import FIXTURES
from tests.forecast_helpers import small_config, synthetic_dataset

NO_COST = CostModel(0.0, 0.0, 0.0)
IDX = pd.bdate_range("2024-01-01", periods=4)


def _bars(opens, closes) -> pd.DataFrame:
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)])


# --- engine ---------------------------------------------------------------------------


def test_engine_matches_hand_calculation() -> None:
    bars = _bars([100, 102, 99, 104], [101, 100, 103, 105])
    targets = pd.Series([1.0, np.nan, 0.5, np.nan], index=IDX)
    costs = CostModel(0.001, 0.0005, 0.0)
    res = run_backtest(bars, targets, costs, "t")
    # day1: flat overnight; buy 1.0 at 102 (cost .0015); 102 -> 100
    e1 = (1 - 0.0015) * (100 / 102)
    # day2: long overnight 100 -> 99; hold; 99 -> 103
    e2 = e1 * (99 / 100) * (103 / 99)
    # day3: long overnight 103 -> 104; sell 0.5 (cost .00075); half 104 -> 105
    e3 = e2 * (104 / 103) * (1 - 0.00075) * (1 + 0.5 * (105 / 104 - 1))
    assert res.equity.to_numpy() == pytest.approx([1.0, e1, e2, e3], rel=1e-12)
    assert res.frame["position"].tolist() == [0.0, 1.0, 1.0, 0.5]
    assert res.frame["trade"].tolist() == [0.0, 1.0, 0.0, -0.5]


def test_signal_trades_at_the_next_open_not_the_same_close() -> None:
    # a target decided at the close of a huge up-day must not earn that day's move
    bars = _bars([100, 100, 200, 200], [100, 200, 200, 200])
    targets = pd.Series([np.nan, 1.0, np.nan, np.nan], index=IDX)
    res = run_backtest(bars, targets, NO_COST, "t")
    assert res.equity.iloc[1] == 1.0 and res.equity.iloc[-1] == pytest.approx(1.0)
    assert res.frame["position"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_roll_cost_only_when_holding_on_roll_days() -> None:
    bars = _bars([100] * 4, [100] * 4)
    costs = CostModel(0.0, 0.0, 0.001)
    rolls = pd.Series([False, False, True, False], index=IDX)
    held = run_backtest(
        bars, pd.Series([1.0, np.nan, np.nan, np.nan], index=IDX), costs, "h", rolls
    )
    flat = run_backtest(
        bars, pd.Series([0.0, np.nan, np.nan, np.nan], index=IDX), costs, "f", rolls
    )
    assert held.frame["cost"].tolist() == [0.0, 0.0, 0.001, 0.0]
    assert flat.frame["cost"].sum() == 0.0


def test_targets_are_clipped_to_unit_gross() -> None:
    bars = _bars([100] * 4, [100] * 4)
    res = run_backtest(bars, pd.Series([3.0, -5.0, np.nan, np.nan], index=IDX), NO_COST, "t")
    assert res.frame["position"].tolist() == [0.0, 1.0, -1.0, -1.0]


def test_buy_and_hold_enters_at_the_second_open() -> None:
    bars = _bars([100, 110, 120, 130], [105, 115, 125, 140])
    res = buy_and_hold(bars, NO_COST)
    assert res.equity.iloc[-1] == pytest.approx(140 / 110)


def _raw(opens, closes, reasons) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame(
        {"open": opens, "close": closes, "mask_reason": reasons}, index=idx, dtype=object
    ).astype({"open": float, "close": float})


def test_pnl_bars_credit_only_what_a_held_contract_earned() -> None:
    raw = _raw(
        [100, 101, 80, 81, 82],
        [100, 102, 82, 83, 84],
        ["missing", "", "roll", "", ""],  # day 2: splice in the overnight gap (102 -> 80)
    )
    b = pnl_bars(raw)
    on = (b["open"] / b["close"].shift(1)).to_numpy()
    intra = (b["close"] / b["open"]).to_numpy()
    assert on[1] == pytest.approx(101 / 100) and intra[1] == pytest.approx(102 / 101)
    assert on[2] == 1.0 and intra[2] == pytest.approx(82 / 80)  # splice not credited
    assert on[3] == pytest.approx(81 / 82) and intra[3] == pytest.approx(83 / 81)
    assert b["zeroed"].tolist() == ["", "", "overnight", "", ""]


def test_exit_on_a_masked_day_does_not_depend_on_its_close() -> None:
    def exit_return(close_on_roll_day: float) -> float:
        raw = _raw(
            [100, 101, 80, 81], [100, 102, close_on_roll_day, 83], ["missing", "", "roll", ""]
        )
        t = pd.Series([1.0, 0.0, np.nan, np.nan], index=raw.index)  # exit at the roll-day open
        return run_backtest(pnl_bars(raw), t, NO_COST, "t").frame["ret"].iloc[2]

    assert exit_return(82) == exit_return(95) == 0.0


def test_nonpositive_prices_earn_nothing() -> None:
    raw = _raw(
        [20, 18, -5, -30, 10, 11],
        [20, 17, -37.6, 10, 11, 12],
        ["missing", "", "nonpositive_price", "nonpositive_price", "", ""],
    )
    b = pnl_bars(raw)
    res = run_backtest(b, pd.Series([1.0] + [np.nan] * 5, index=raw.index), NO_COST, "t")
    assert res.frame["ret"].iloc[2] == res.frame["ret"].iloc[3] == 0.0
    assert b["zeroed"].tolist()[2:4] == ["both", "both"]
    assert np.isfinite(res.equity).all() and (res.equity > 0).all()


def test_benchmark_pays_roll_costs_like_an_always_long_strategy() -> None:
    bars = _bars([100] * 4, [100] * 4)
    rolls = pd.Series([False, False, True, False], index=IDX)
    costs = CostModel(0.001, 0.0, 0.002)
    bench = buy_and_hold(bars, costs, rolls)
    strat = run_backtest(bars, pd.Series([1.0] + [np.nan] * 3, index=IDX), costs, "s", rolls)
    assert bench.frame["cost"].sum() == pytest.approx(strat.frame["cost"].sum())
    assert bench.frame["cost"].iloc[2] == pytest.approx(0.002)


# --- metrics --------------------------------------------------------------------------


def test_metrics_on_a_known_path() -> None:
    bars = _bars([100, 100, 110, 99], [100, 110, 99, 99])
    res = run_backtest(bars, pd.Series([1.0, np.nan, np.nan, 0.0], index=IDX), NO_COST, "t")
    m = summarize(res)
    assert m.total_return == pytest.approx(0.99 - 1)
    assert m.max_drawdown == pytest.approx(99 / 110 - 1)
    assert m.trades == 1 and m.exposure == pytest.approx(1.0)
    assert m.hit_rate == pytest.approx(1 / 3)
    assert m.sharpe_ci95 is not None and m.sharpe_ci95[0] < m.sharpe < m.sharpe_ci95[1]


def test_sharpe_ci_uses_the_per_period_sharpe() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=501)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(idx))))
    bars = pd.DataFrame({"open": np.r_[100, c[:-1]], "close": c}, index=idx)
    m = summarize(buy_and_hold(bars, NO_COST))
    n = m.days
    sr_d = m.sharpe / np.sqrt(252)
    se = np.sqrt((1 + 0.5 * sr_d**2) / n) * np.sqrt(252)
    assert m.sharpe_ci95 == pytest.approx((m.sharpe - 1.96 * se, m.sharpe + 1.96 * se))


def _random_walk_bars(rng: np.random.Generator, n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": o, "close": c}, index=idx)


def test_edge_pvalue_detects_foresight() -> None:
    bars = _random_walk_bars(np.random.default_rng(7), 1500)
    bench = buy_and_hold(bars, NO_COST)
    up_next = (bars["close"] / bars["open"] > 1).astype(float).shift(-1)  # peeks: test only
    assert edge_pvalue(run_backtest(bars, up_next, NO_COST, "oracle"), bench, 20) < 0.001
    assert np.isnan(edge_pvalue(bench, bench, 20))


def test_edge_pvalue_size_on_coin_flips() -> None:
    """Random signals should look significant about alpha of the time, not more."""
    rng = np.random.default_rng(11)
    pvals = []
    for _ in range(100):
        bars = _random_walk_bars(rng, 750)
        coin = pd.Series(rng.integers(0, 2, len(bars)).astype(float), index=bars.index)
        coin = coin.where(np.arange(len(bars)) % 5 == 0)  # a decision every 5 sessions
        pvals.append(
            edge_pvalue(run_backtest(bars, coin, NO_COST, "c"), buy_and_hold(bars, NO_COST), 20)
        )
    assert np.mean(np.asarray(pvals) < 0.05) <= 0.10


def test_flat_strategy_has_no_sharpe_and_no_trades() -> None:
    bars = _bars([100, 101, 102, 103], [101, 102, 103, 104])
    m = summarize(run_backtest(bars, pd.Series(0.0, index=IDX), NO_COST, "flat"))
    assert m.total_return == 0.0 and m.sharpe is None and m.trades == 0
    json.dumps(m.model_dump())  # FiniteModel: NaN -> None, serializable


def test_compare_uses_the_same_window() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame({"open": np.arange(100, 106.0), "close": np.arange(100, 106.0)}, index=idx)
    targets = pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan], index=idx)
    strat, bench = compare(bars, targets, pd.Series(False, index=idx), NO_COST, "s")
    assert strat.frame.index[0] == bench.frame.index[0] == idx[2]
    assert strat.equity.iloc[-1] == pytest.approx(bench.equity.iloc[-1])


def test_compare_rejects_a_signal_without_targets() -> None:
    bars = _bars([100] * 4, [100] * 4)
    with pytest.raises(ValueError, match="never produced"):
        compare(bars, pd.Series(np.nan, index=IDX), pd.Series(False, index=IDX), NO_COST, "s")


# --- signal causality -----------------------------------------------------------------

PRICES = pd.read_csv(FIXTURES / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
CUTS = [230, 300, 377, 450, 519]
SMA = SMASettings(fast=20, slow=60)


def _level(d: pd.DataFrame) -> pd.Series:
    return np.exp(np.log(d["close"]).diff().fillna(0).cumsum())


def test_sma_signal_is_causal() -> None:
    assert_causal(lambda d: sma_crossover(_level(d), SMA, True).to_frame("t"), PRICES, CUTS)


def test_leaky_signal_is_caught() -> None:
    def peeking(d: pd.DataFrame) -> pd.DataFrame:  # decides today on tomorrow's return
        nxt = _level(d).pct_change().shift(-1)
        return pd.DataFrame({"t": np.sign(nxt)}, index=d.index)

    with pytest.raises(LeakageError):
        assert_causal(peeking, PRICES, CUTS)


def test_vol_targeting_is_causal() -> None:
    def build(d: pd.DataFrame) -> pd.DataFrame:
        r = np.log(d["close"]).diff()
        t = sma_crossover(_level(d), SMA, False)
        return vol_targeted(t, r, 0.2, 1.0).to_frame("t")

    assert_causal(build, PRICES, CUTS)


def test_sma_is_long_only_by_default() -> None:
    t = sma_crossover(_level(PRICES), SMA, long_short=False).dropna()
    assert set(t.unique()) <= {0.0, 1.0} and t.index[0] == PRICES.index[SMA.slow - 1]


def test_forecast_targets_only_at_walk_forward_origins(cfg: AppConfig) -> None:
    small = small_config(cfg)
    ds = synthetic_dataset(small, n=600)
    fs = small.backtest.forecast.model_copy(
        update={"model": "drift", "members": ["naive", "drift"]}
    )
    targets, ev = forecast_targets(ds, small, fs, long_short=False)
    decided = targets.dropna()
    origins = ds.dates[ev.positions[fs.horizon]]
    assert set(decided.index) <= set(origins) and len(decided) > 0
    assert decided.index[0] >= ds.dates[small.forecasting.walk_forward.min_train_days - 1]
    assert set(decided.unique()) <= {0.0, 1.0}


# --- agent backtest helpers -----------------------------------------------------------


def test_decision_positions_spacing_and_cap() -> None:
    assert decision_positions(list(range(0, 100, 5)), 20, 3) == [40, 60, 80]
    assert decision_positions([0, 5, 30, 31, 55], 20, 10) == [0, 30, 55]


def test_rebase_factor_fails_closed_and_never_looks_ahead() -> None:
    close = pd.Series([50.0, -3.0, np.nan, 80.0])
    assert rebase_factor(close, 0) == pytest.approx(2.0)
    assert rebase_factor(close, 2) == pytest.approx(2.0)  # last positive at or before, not 80
    with pytest.raises(ValueError):
        rebase_factor(pd.Series([np.nan, -1.0, 5.0]), 1)


def test_cutoff_notes(cfg: AppConfig) -> None:
    dates = [pd.Timestamp("2025-06-02"), pd.Timestamp("2026-06-01")]
    notes, models = cutoff_notes(cfg, dates)
    assert models and all(m.startswith("claude-") for m in models.values())
    assert all("training cutoff" in n for n in notes)
    roles = cfg.models.roles.copy()
    roles["analyst"] = roles["analyst"].model_copy(update={"training_cutoff": date(2026, 1, 1)})
    cfg2 = cfg.model_copy(update={"models": cfg.models.model_copy(update={"roles": roles})})
    notes2, _ = cutoff_notes(cfg2, dates)
    assert any("1 of 2 decisions" in n for n in notes2)


def test_forecast_component_needs_a_significant_settled_edge() -> None:
    row = {"p_up": 0.7}
    assert _forecast_component(None, row, 0.05)[1] is False
    bad = {"pinball_skill_vs_random_walk": -0.01, "dm_p_value": 0.01}
    assert _forecast_component(bad, row, 0.05)[1] is False
    weak = {"pinball_skill_vs_random_walk": 0.02, "dm_p_value": 0.2}
    assert _forecast_component(weak, row, 0.05)[1] is False
    good = {"pinball_skill_vs_random_walk": 0.02, "dm_p_value": 0.01}
    score, used, _ = _forecast_component(good, row, 0.05)
    assert used and score == pytest.approx(0.4)


def test_past_skill_uses_only_settled_origins(cfg: AppConfig) -> None:
    small = small_config(cfg)
    ds = synthetic_dataset(small, n=600)
    fs = small.backtest.forecast.model_copy(
        update={"model": "drift", "members": ["naive", "drift"]}
    )
    _, ev = forecast_targets(ds, small, fs, long_short=False)
    h = fs.horizon
    pos = int(ev.positions[h][-1])
    before = past_skill(ev, "drift", pos, small.forecasting)[h]
    # poison every label not final by `pos` (its roll mask looks up to embargo_days ahead):
    # the skill must not move
    unsettled = ev.positions[h] + h + small.forecasting.walk_forward.embargo_days > pos
    y = ev.realized[h].copy()
    y.iloc[np.flatnonzero(unsettled)] = 1e6
    poisoned = type(ev)(**{**ev.__dict__, "realized": {**ev.realized, h: y}})
    assert past_skill(poisoned, "drift", pos, small.forecasting)[h] == before
    assert before is not None and before["forecasts_scored"] == int((~unsettled).sum())


# --- anonymized payloads --------------------------------------------------------------

_WORDS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "wti",
    "brent",
    "crude",
    "oil",
    "gas",
    "henry",
    "ttf",
    "nymex",
    "ice",
    "cme",
    "eia",
    "cftc",
    "cushing",
    "barrel",
    "barrels",
    "bbl",
    "usd",
    "gasoline",
    "distillate",
    "mmbtu",
    "crack",
    "opec",
]
LEAKY = re.compile(
    r"(19|20)\d\d-\d\d|(?<![\d.])(19|20)\d\d(?![\d.])|\$|cl=f|(?<![a-z])("
    + "|".join(_WORDS)
    + r")(?![a-z])",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def cl_snapshot() -> Snapshot:
    import tempfile
    from pathlib import Path

    from fa.analytics.snapshot import build_snapshot
    from fa.config import load_config
    from tests.integration.test_snapshot import _service

    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(load_config(), Path(tmp))
        return build_snapshot(svc, "CL=F", as_of=date(2026, 6, 1), include_curve=False)


def _anon_ctx(cfg: AppConfig, snap: Snapshot, tmp_path) -> RunContext:
    ctx = RunContext(
        cfg=cfg,
        svc=None,  # type: ignore[arg-type]
        symbol="ASSET",
        as_of=date(2026, 6, 1),
        scratchpad=Scratchpad(tmp_path, run_id="anon"),
    )
    ctx.cache["anon"] = AnonState(
        snapshot=snap,
        factor=100.0 / snap.price.last_close,
        forecast_rows={5: {"q0.1": -0.03, "q0.5": 0.001, "q0.9": 0.03, "p_up": 0.52}},
        past_skill={
            5: {"forecasts_scored": 40, "pinball_skill_vs_random_walk": -0.01, "dm_p_value": 0.6}
        },
        sessions_to_roll=7,
    )
    return ctx


def _leaks(payload) -> list[str]:
    """Keys and strings matching LEAKY, plus integers that look like calendar years."""
    if isinstance(payload, dict):
        return [x for k, v in payload.items() for x in _leaks(k) + _leaks(v)]
    if isinstance(payload, list):
        return [x for v in payload for x in _leaks(v)]
    if isinstance(payload, str):
        return [m.group(0) for m in LEAKY.finditer(payload)]
    if isinstance(payload, int) and not isinstance(payload, bool) and 1980 <= payload <= 2100:
        return [str(payload)]
    return []


def _call(ctx: RunContext, name: str, allow_error: bool = False) -> dict:
    res = build_anon_registry().call(ctx, "technical_analyst", name, {})
    assert allow_error or not res.is_error, res.payload
    return res.payload


def test_anonymized_payloads_hide_names_dates_and_units(
    cfg: AppConfig, cl_snapshot, tmp_path
) -> None:
    ctx = _anon_ctx(cfg, cl_snapshot, tmp_path)
    for name, (desc, _) in ANON_TOOLS.items():
        payload = _call(ctx, name, allow_error=name == "processing_margin")  # no crack here
        assert not _leaks(payload), f"{name} leaks {_leaks(payload)}: {payload}"
        assert not LEAKY.search(desc), f"{name} description leaks: {desc}"
    # the scratchpad records exactly the anonymized payloads (what the claim checker sees)
    logged = [json.loads(line) for line in ctx.scratchpad.path.read_text().splitlines()]
    assert all(not _leaks(e.get("result", {})) for e in logged)


def test_prices_are_rebased(cfg: AppConfig, cl_snapshot, tmp_path) -> None:
    out = _call(_anon_ctx(cfg, cl_snapshot, tmp_path), "price_technicals")
    assert out["price_index"]["last"] == pytest.approx(100.0)
    raw = cl_snapshot.price.last_close
    for k in ("high_52w", "low_52w"):
        assert out["price_index"][k] == pytest.approx(
            getattr(cl_snapshot.price, k) * 100 / raw, rel=1e-5
        )


def test_inventories_are_ratios_only(cfg: AppConfig, cl_snapshot, tmp_path) -> None:
    out = _call(_anon_ctx(cfg, cl_snapshot, tmp_path), "inventories")
    kinds = [s["kind"] for s in out["series"] if s["available"]]
    assert kinds and set(kinds) <= {"stocks", "supply", "processing rate", "other"}
    for s in out["series"]:
        assert set(s) == {"name", "kind", "available", "vs_5y_average", "position_in_5y_band"}
    # production and utilization are never presented as inventories
    assert "stocks" in kinds and ("supply" in kinds or "processing rate" in kinds)


def test_payload_shape_does_not_depend_on_the_asset(cfg: AppConfig, cl_snapshot, tmp_path) -> None:
    bare = cl_snapshot.model_copy(update={"inventories": [], "cot": None, "crack": None})

    def shape(v):
        if isinstance(v, dict):
            return {k: shape(x) for k, x in v.items() if k != "result_id"}
        if isinstance(v, list):
            return [shape(x) for x in v]
        return None

    full_ctx, bare_ctx = _anon_ctx(cfg, cl_snapshot, tmp_path), _anon_ctx(cfg, bare, tmp_path)
    for name in ANON_TOOLS:
        a = build_anon_registry().call(full_ctx, "x", name, {})
        b = build_anon_registry().call(bare_ctx, "x", name, {})
        assert a.is_error == b.is_error and shape(a.payload) == shape(b.payload), name


def test_anon_agents_only_get_anon_tools() -> None:
    assert {t for ts in ANON_AGENT_TOOLS.values() for t in ts} <= set(ANON_TOOLS)
    assert "news" not in ANON_TOOLS and "macro" not in ANON_TOOLS


def test_accounting_notes_flag_negative_prices_while_invested() -> None:
    from fa.backtest.service import accounting_notes

    raw = _raw(
        [20, 18, -5, -30, 10, 11],
        [20, 17, -37.6, 10, 11, 12],
        ["missing", "", "nonpositive_price", "nonpositive_price", "roll", ""],
    )
    bars = pnl_bars(raw)
    bench = buy_and_hold(bars, NO_COST)
    flat = run_backtest(bars, pd.Series(0.0, index=raw.index), NO_COST, "flat")
    notes = accounting_notes(bars, flat, bench)
    warning = next(n for n in notes if n.startswith("WARNING"))
    assert "2024-01-03, 2024-01-04" in warning
    assert "strategy flat, buy-and-hold invested" in warning
    assert any("1 roll sessions" in n for n in notes)
