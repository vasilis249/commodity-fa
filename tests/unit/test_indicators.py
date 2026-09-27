"""Indicators vs TA-Lib reference values (tests/fixtures/indicator_reference.csv),
the StockCharts RSI textbook example, and closed-form cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fa.analytics import indicators as ind
from tests.conftest import FIXTURES

PRICES = pd.read_csv(FIXTURES / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
REF = pd.read_csv(FIXTURES / "indicator_reference.csv", index_col=0, parse_dates=True)
C, H, L = PRICES["close"], PRICES["high"], PRICES["low"]


def _same(ours: pd.Series, ref: pd.Series, start: int = 0, tol: float = 1e-8) -> None:
    assert (ours.notna() == ref.notna()).iloc[start:].all(), "warm-up (NaN) pattern differs"
    np.testing.assert_allclose(ours.iloc[start:], ref.iloc[start:], atol=tol, equal_nan=True)


@pytest.mark.parametrize(
    ("name", "compute"),
    [
        ("sma20", lambda: ind.sma(C, 20)),
        ("ema20", lambda: ind.ema(C, 20)),
        ("rsi14", lambda: ind.rsi(C, 14)),
        ("bb_upper", lambda: ind.bollinger(C, 20, 2)["bb_upper"]),
        ("bb_mid", lambda: ind.bollinger(C, 20, 2)["bb_mid"]),
        ("bb_lower", lambda: ind.bollinger(C, 20, 2)["bb_lower"]),
        ("trange", lambda: ind.true_range(H, L, C)),
        ("atr14", lambda: ind.atr(H, L, C, 14)),
        ("stddev20", lambda: C.rolling(20).std(ddof=0)),
    ],
)
def test_matches_talib(name: str, compute) -> None:
    _same(compute(), REF[name])


@pytest.mark.parametrize("col", ["macd", "macd_signal", "macd_hist"])
def test_macd_matches_talib_after_warmup(col: str) -> None:
    # TA-Lib aligns the fast EMA's seed with the slow one; seeds wash out in ~100 bars
    ours = ind.macd(C)[col]
    np.testing.assert_allclose(ours.iloc[150:], REF[col].iloc[150:], atol=1e-7)


def test_rsi_stockcharts_textbook_example() -> None:
    """ChartSchool RSI table; their spreadsheet rounds intermediate averages (±0.1)."""
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
              46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]  # fmt: skip
    published = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]
    out = ind.rsi(pd.Series(closes), 14).dropna().to_numpy()
    np.testing.assert_allclose(out, published, atol=0.1)


def test_closed_form_cases() -> None:
    flat = pd.Series([50.0] * 40)
    assert ind.ema(flat, 10).dropna().eq(50).all()
    assert ind.rsi(flat).dropna().eq(50).all()
    bb = ind.bollinger(flat)
    assert (bb["bb_upper"] - bb["bb_lower"]).dropna().eq(0).all()
    assert bb["bb_pct_b"].isna().all()
    up = pd.Series(np.arange(1.0, 41.0))
    assert ind.rsi(up).dropna().eq(100).all()
    assert ind.sma(up, 5).iloc[4] == 3.0
    rng = pd.Series([10.0] * 30)
    assert ind.atr(rng + 1, rng - 1, rng, 14).dropna().eq(2).all()


def test_gaps_are_rejected() -> None:
    s = pd.Series([1.0, 2.0, np.nan, 4.0] + [5.0] * 20)
    with pytest.raises(ValueError, match="gaps"):
        ind.ema(s, 5)
    lead = pd.Series([np.nan, np.nan] + [1.0] * 20)
    assert ind.ema(lead, 5).first_valid_index() == 6  # leading NaNs are fine


@pytest.mark.parametrize(
    "fn",
    [
        lambda c: ind.sma(c, 20),
        lambda c: ind.ema(c, 20),
        lambda c: ind.rsi(c, 14),
        lambda c: ind.macd(c)["macd_hist"],
        lambda c: ind.bollinger(c)["bb_pct_b"],
        lambda c: ind.realized_vol(ind.log_returns(c), 20),
        lambda c: ind.drawdown(c),
        lambda c: ind.rolling_percentile(c, 60),
    ],
)
def test_no_look_ahead(fn) -> None:
    """Values up to t must not change when later bars are appended."""
    cut = 300
    np.testing.assert_allclose(fn(C.iloc[:cut]), fn(C).iloc[:cut], equal_nan=True)


def test_realized_vol_and_drawdown_known_values() -> None:
    r = pd.Series([0.01, -0.01] * 10)
    expected = np.std([0.01, -0.01] * 10, ddof=1) * np.sqrt(252)
    assert ind.realized_vol(r, 20).iloc[-1] == pytest.approx(expected)
    level = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0])
    assert list(ind.drawdown(level)) == pytest.approx([0, 0, -0.25, 110 / 120 - 1, 0])


def test_realized_vol_skips_masked_days() -> None:
    r = pd.Series([0.01, -0.01] * 10)
    r.iloc[5] = np.nan
    assert np.isfinite(ind.realized_vol(r, 20).iloc[-1])


def test_rolling_percentile() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ind.rolling_percentile(x, 5).iloc[-1] == 100.0
    assert ind.rolling_percentile(pd.Series([5.0, 4, 3, 2, 1]), 5).iloc[-1] == 0.0
    assert ind.rolling_percentile(pd.Series([1.0, 3, 2]), 3).iloc[-1] == 50.0


def test_log_returns_mask_nonpositive() -> None:
    r = ind.log_returns(pd.Series([10.0, 11.0, -5.0, 12.0]))
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert r.iloc[2:].isna().all()
