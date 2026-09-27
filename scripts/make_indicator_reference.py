"""Regenerate tests/fixtures/indicator_reference.csv from TA-Lib (the reference oracle).

TA-Lib is NOT a project dependency. Run with:
    uv run --with "ta-lib>=0.6" python scripts/make_indicator_reference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import talib

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

prices = pd.read_csv(FIX / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
o, h, lo, c = (prices[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))

ref = pd.DataFrame(index=prices.index)
ref["sma20"] = talib.SMA(c, 20)
ref["ema20"] = talib.EMA(c, 20)
ref["rsi14"] = talib.RSI(c, 14)
ref["macd"], ref["macd_signal"], ref["macd_hist"] = talib.MACD(c, 12, 26, 9)
ref["bb_upper"], ref["bb_mid"], ref["bb_lower"] = talib.BBANDS(c, 20, 2.0, 2.0, 0)
ref["trange"] = talib.TRANGE(h, lo, c)
ref["atr14"] = talib.ATR(h, lo, c, 14)
ref["stddev20"] = talib.STDDEV(c, 20, 1.0)  # population std
ref.to_csv(FIX / "indicator_reference.csv", float_format="%.10f")
print(
    f"talib {talib.__version__}: wrote {len(ref)} rows, {np.isfinite(ref.to_numpy()).sum()} values"
)
