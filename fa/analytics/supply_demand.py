"""Supply/demand analytics: inventories vs seasonal norms, COT positioning, cracks.

Pure functions. Inputs are the data layer's frames; point-in-time alignment happens
before (fa.data.pit.pit_join) or via `available_at`, never here.
"""

from __future__ import annotations

import pandas as pd

from fa.analytics.indicators import rolling_percentile
from fa.analytics.seasonality import seasonal_band

BARRELS_PER_GALLON_CONTRACT = 42  # HO/RB quote USD/gal; 42 gal per barrel


def inventory_vs_seasonal(weekly: pd.Series, years: int = 5) -> pd.DataFrame:
    """Weekly level vs the prior-`years` seasonal band, plus week-over-week surprise.

    Columns: value, avg5, min5, max5, n_years, dev (value - avg5), dev_pct,
    band_pos (0 = at 5y min, 1 = at 5y max), wow (week-over-week change),
    wow_seasonal (average same-week change in prior years), wow_vs_seasonal.
    """
    band = seasonal_band(weekly, years)
    out = pd.DataFrame({"value": weekly}, index=weekly.index)
    out["avg5"], out["min5"], out["max5"], out["n_years"] = (
        band["avg"],
        band["min"],
        band["max"],
        band["n_years"],
    )
    out["dev"] = out["value"] - out["avg5"]
    out["dev_pct"] = out["dev"] / out["avg5"]
    width = out["max5"] - out["min5"]
    out["band_pos"] = ((out["value"] - out["min5"]) / width).where(width > 0)
    out["wow"] = weekly.diff()
    out["wow_seasonal"] = seasonal_band(out["wow"].dropna(), years)["avg"].reindex(out.index)
    out["wow_vs_seasonal"] = out["wow"] - out["wow_seasonal"]
    return out


def cot_positioning(cot: pd.DataFrame, percentile_weeks: int = 156) -> pd.DataFrame:
    """Managed-money and commercial net positions with a trailing percentile rank.

    Commercials = producers/merchants + swap dealers.
    """
    out = pd.DataFrame(index=cot.index)
    out["mm_net"] = cot["mm_long"] - cot["mm_short"]
    out["mm_net_pct_oi"] = out["mm_net"] / cot["open_interest"]
    out["mm_net_change_1w"] = out["mm_net"].diff()
    out["mm_net_pctile"] = rolling_percentile(
        out["mm_net"], percentile_weeks, min_periods=percentile_weeks // 2
    )
    out["commercial_net"] = (cot["prod_long"] + cot["swap_long"]) - (
        cot["prod_short"] + cot["swap_short"]
    )
    out["open_interest"] = cot["open_interest"]
    if "available_at" in cot:
        out["available_at"] = cot["available_at"]
    return out


def crack_spread_321(cl: pd.Series, rb: pd.Series, ho: pd.Series) -> pd.Series:
    """3-2-1 crack in USD/bbl: (2 x RBOB + 1 x ULSD - 3 x WTI) / 3.

    RB and HO are quoted in USD/gal. Uses front months, whose roll dates differ
    (CL around the 20th, RB/HO at month end), so values around rolls mix contract
    months; the snapshot flags that.
    """
    df = pd.concat({"cl": cl, "rb": rb, "ho": ho}, axis=1).dropna()
    g = BARRELS_PER_GALLON_CONTRACT
    return ((2 * df["rb"] * g + df["ho"] * g - 3 * df["cl"]) / 3).rename("crack_321")


def percentile_last(x: pd.Series, window: int) -> float:
    """Percentile rank (0..100) of the last value within the last `window` values."""
    tail = x.dropna().iloc[-window:]
    if len(tail) < 2:
        return float("nan")
    last = tail.iloc[-1]
    return float(100 * ((tail < last).sum() + 0.5 * ((tail == last).sum() - 1)) / (len(tail) - 1))
