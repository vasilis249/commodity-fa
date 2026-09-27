"""Technical indicators. Pure, causal (value at t uses data up to t), tested vs TA-Lib.

Conventions (match TA-Lib / StockCharts):
- EMA and Wilder averages are seeded with the simple average of the first `n` values.
- RSI and ATR use Wilder smoothing (alpha = 1/n); ATR's seed skips the first bar's
  true range (it has no previous close).
- Bollinger bands use the population standard deviation.
Inputs must have no gaps after their first valid value (use `close_adj`, not `close`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _values(x: pd.Series) -> tuple[np.ndarray, int]:
    """Float array and the position of its first valid value; reject interior gaps."""
    arr = x.to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(arr))
    if valid.size == 0:
        return arr, len(arr)
    start = int(valid[0])
    if np.isnan(arr[start:]).any():
        raise ValueError(f"{x.name or 'series'} has gaps after its first value")
    return arr, start


def _recursive_average(x: pd.Series, n: int, alpha: float) -> pd.Series:
    arr, start = _values(x)
    out = np.full(len(arr), np.nan)
    if len(arr) - start >= n:
        seed = start + n - 1
        out[seed] = arr[start : seed + 1].mean()
        for i in range(seed + 1, len(arr)):
            out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return pd.Series(out, index=x.index, name=x.name)


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return _recursive_average(x, n, 2.0 / (n + 1))


def wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA)."""
    return _recursive_average(x, n, 1.0 / n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain = wilder(diff.clip(lower=0), n)
    loss = wilder((-diff).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    out = out.where(loss != 0, 100.0)
    out = out.where(~((gain == 0) & (loss == 0)), 50.0)
    return out.where(gain.notna()).rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal and histogram.

    Each EMA is seeded independently from the first bar. TA-Lib instead aligns the
    fast EMA's start with the slow one, so values differ during warm-up and converge
    within ~100 bars.
    """
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "bb_pct_b": ((close - lower) / width).where(width > 0),
            "bb_bandwidth": (width / mid).where(mid != 0),
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[:1] = np.nan  # no previous close on the first bar (TA-Lib convention)
    return tr.rename("true_range")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n).rename("atr")


def log_returns(close: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close / close.shift(1))
    return r.where((close > 0) & (close.shift(1) > 0)).rename("log_return")


def realized_vol(
    returns: pd.Series, n: int = 20, periods: int = TRADING_DAYS, min_fraction: float = 0.8
) -> pd.Series:
    """Annualized rolling standard deviation of (log) returns.

    Masked returns (NaN) are skipped; a window needs `min_fraction` of its values.
    """
    minp = max(2, int(np.ceil(n * min_fraction)))
    return (returns.rolling(n, min_periods=minp).std(ddof=1) * np.sqrt(periods)).rename("vol")


def drawdown(level: pd.Series) -> pd.Series:
    """Fractional distance below the running peak (0 at a new high, negative below)."""
    return (level / level.cummax() - 1).rename("drawdown")


def rolling_percentile(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank (0..100) of each value within its trailing window (causal)."""

    def rank_last(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.nan
        last = a[-1]
        return 100.0 * ((a < last).sum() + 0.5 * ((a == last).sum() - 1)) / max(a.size - 1, 1)

    return x.rolling(window, min_periods=min_periods or window).apply(rank_last, raw=True)


def pct_change_over(level: pd.Series, n: int) -> pd.Series:
    """Simple return over the last n bars."""
    return (level / level.shift(n) - 1).rename(f"ret_{n}")
