from __future__ import annotations

import numpy as np
import pandas as pd

from fa.analytics import technicals as tech


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_trend_regimes() -> None:
    up = _series(np.linspace(50, 150, 400))
    down = _series(np.linspace(150, 50, 400))
    assert tech.trend_regime(up).iloc[-1] == "uptrend"
    assert tech.trend_regime(down).iloc[-1] == "downtrend"
    assert tech.trend_regime(up).iloc[100] == "insufficient_data"
    flat = _series(100 + np.sin(np.arange(400) / 3))
    assert tech.trend_regime(flat).iloc[-1] == "range"


def test_vol_regime_flags_a_vol_spike() -> None:
    rng = np.random.default_rng(3)
    r = _series(np.r_[rng.normal(0, 0.01, 800), rng.normal(0, 0.05, 25)])
    regime, pct = tech.vol_regime(r)
    assert regime.iloc[-1] == "extreme" and pct.iloc[-1] > 95
    assert regime.iloc[10] == "insufficient_data"


def test_pivots_are_confirmed_k_bars_later() -> None:
    values = [1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1, 2, 3]
    s = _series(values)
    piv = tech.swing_pivots(s, s, k=5)
    top = piv[piv["kind"] == "resistance"]
    assert list(top.index) == [s.index[5]]
    assert top["confirmed_at"].iloc[0] == s.index[10]  # usable only 5 bars later
    # the last k bars can never be pivots yet (no look-ahead)
    assert (piv.index <= s.index[-6]).all()


def test_support_resistance_around_last_close() -> None:
    zig = np.tile(np.r_[np.linspace(90, 110, 11), np.linspace(110, 90, 11)[1:-1]], 12)
    s = _series(np.r_[zig, np.full(10, 100.0)])
    levels = tech.support_resistance(s, s, s, atr_value=1.0)
    sup = [lv for lv in levels if lv.kind == "support"]
    res = [lv for lv in levels if lv.kind == "resistance"]
    assert sup and res
    assert sup[0].price == 92  # the last swing low before the flat stretch is nearest
    assert sup[1].price == 90 and sup[1].touches > 1  # repeated lows merge into one level
    assert res[0].price == 110 and res[0].touches > 1
