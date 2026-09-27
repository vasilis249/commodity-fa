"""Trend and volatility regimes and support/resistance levels. Pure and causal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fa.analytics.indicators import realized_vol, rolling_percentile, sma

Trend = Literal["uptrend", "downtrend", "range", "insufficient_data"]
VolRegime = Literal["low", "normal", "high", "extreme", "insufficient_data"]


def trend_regime(
    close: pd.Series, fast: int = 50, slow: int = 200, slope_bars: int = 20
) -> pd.Series:
    """Uptrend: close > SMA(fast) > SMA(slow) and SMA(slow) rising; downtrend: mirror."""
    f, s = sma(close, fast), sma(close, slow)
    rising = s > s.shift(slope_bars)
    falling = s < s.shift(slope_bars)
    out = pd.Series("range", index=close.index, dtype="object")
    out[(close > f) & (f > s) & rising] = "uptrend"
    out[(close < f) & (f < s) & falling] = "downtrend"
    out[s.shift(slope_bars).isna()] = "insufficient_data"
    return out.rename("trend")


def vol_regime(returns: pd.Series, n: int = 20, lookback: int = 756) -> tuple[pd.Series, pd.Series]:
    """(regime, percentile): current n-day vol vs its own trailing `lookback` history."""
    vol = realized_vol(returns, n)
    pct = rolling_percentile(vol, lookback, min_periods=lookback // 3)
    regime = pd.Series("normal", index=returns.index, dtype="object")
    regime[pct < 25] = "low"
    regime[pct > 75] = "high"
    regime[pct > 95] = "extreme"
    regime[pct.isna()] = "insufficient_data"
    return regime.rename("vol_regime"), pct.rename("vol_percentile")


@dataclass(frozen=True)
class Level:
    price: float
    kind: Literal["support", "resistance"]
    touches: int  # pivots merged into this level
    last_seen: pd.Timestamp


def swing_pivots(high: pd.Series, low: pd.Series, k: int = 5) -> pd.DataFrame:
    """Pivot highs/lows: extreme within +-k bars. Confirmed (usable) only k bars later.

    Returns rows indexed by the pivot bar with `price`, `kind` and `confirmed_at`.
    """
    win = 2 * k + 1
    is_high = high == high.rolling(win, center=True, min_periods=win).max()
    is_low = low == low.rolling(win, center=True, min_periods=win).min()
    idx = high.index
    confirmed = pd.Series(idx, index=idx).shift(-k)  # timestamp k bars after the pivot
    rows = [
        *(
            {"date": d, "price": float(high[d]), "kind": "resistance", "confirmed_at": confirmed[d]}
            for d in idx[is_high.fillna(False).to_numpy()]
        ),
        *(
            {"date": d, "price": float(low[d]), "kind": "support", "confirmed_at": confirmed[d]}
            for d in idx[is_low.fillna(False).to_numpy()]
        ),
    ]
    if not rows:
        return pd.DataFrame(columns=["price", "kind", "confirmed_at"])
    return pd.DataFrame(rows).set_index("date").sort_index()


def support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_value: float,
    as_of: pd.Timestamp | None = None,
    lookback: int = 250,
    k: int = 5,
    merge_atr: float = 0.5,
    n_each: int = 2,
) -> list[Level]:
    """Nearest confirmed pivot levels below (support) and above (resistance) the last close.

    Pivots within `merge_atr` x ATR of each other are merged (averaged).
    """
    as_of = as_of or close.index[-1]
    h, lo = high.loc[:as_of].iloc[-lookback:], low.loc[:as_of].iloc[-lookback:]
    last = float(close.loc[:as_of].iloc[-1])
    piv = swing_pivots(h, lo, k)
    piv = piv[piv["confirmed_at"].notna() & (piv["confirmed_at"] <= as_of)]
    levels: list[Level] = []
    for kind, side in (("support", piv["price"] < last), ("resistance", piv["price"] > last)):
        cand = piv[side].sort_values("price", ascending=(kind == "resistance"))
        clusters: list[list[tuple[float, pd.Timestamp]]] = []
        for d, row in cand.iterrows():
            if clusters and abs(row["price"] - clusters[-1][0][0]) <= merge_atr * atr_value:
                clusters[-1].append((row["price"], d))
            else:
                clusters.append([(row["price"], d)])
        for cl in clusters[:n_each]:
            levels.append(
                Level(
                    price=float(np.mean([p for p, _ in cl])),
                    kind=kind,  # type: ignore[arg-type]
                    touches=len(cl),
                    last_seen=max(d for _, d in cl),
                )
            )
    return levels
