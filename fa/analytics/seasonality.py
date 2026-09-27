"""Calendar seasonality of returns and weekly seasonal bands. Pure and causal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fa.analytics.models import FiniteModel


def monthly_returns(log_returns: pd.Series, min_coverage: float = 0.8) -> pd.Series:
    """Sum of daily log returns per calendar month (index: month end).

    Months where fewer than `min_coverage` of the days have a valid (unmasked) return
    are dropped rather than biased toward zero.
    """
    r = log_returns
    grouper = r.index.to_period("M")
    total = r.groupby(grouper).size()
    valid = r.notna().groupby(grouper).sum()
    sums = r.groupby(grouper).sum(min_count=1)
    keep = (valid / total) >= min_coverage
    out = sums[keep]
    out.index = out.index.to_timestamp(how="end").normalize()
    return out.rename("monthly_log_return")


class MonthStats(FiniteModel):
    month: int
    years: int
    mean: float | None  # mean monthly log return
    median: float | None
    hit_rate: float | None  # share of years with a positive month
    t_stat: float | None
    first_year: int | None
    last_year: int | None


def month_stats(monthly: pd.Series, month: int, as_of: pd.Timestamp) -> MonthStats:
    """Stats for calendar `month` using only months that ended before `as_of`."""
    past = monthly[(monthly.index < as_of) & (monthly.index.month == month)].dropna()
    n = len(past)
    sd = float(past.std(ddof=1)) if n > 1 else float("nan")
    return MonthStats(
        month=month,
        years=n,
        mean=float(past.mean()) if n else float("nan"),
        median=float(past.median()) if n else float("nan"),
        hit_rate=float((past > 0).mean()) if n else float("nan"),
        t_stat=float(past.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
        first_year=int(past.index[0].year) if n else None,
        last_year=int(past.index[-1].year) if n else None,
    )


def week_of_year(index: pd.DatetimeIndex) -> pd.Index:
    """ISO week, with week 53 folded into 52 so every year has comparable weeks."""
    return pd.Index(np.minimum(index.isocalendar().week.to_numpy(), 52), name="week")


def seasonal_band(weekly: pd.Series, years: int = 5) -> pd.DataFrame:
    """For each week, the same week's values in the previous `years` years.

    Columns: avg, min, max, n_years (only prior years count, like EIA's 5-year range).
    """
    df = pd.DataFrame(
        {"value": weekly.to_numpy(), "year": weekly.index.year, "week": week_of_year(weekly.index)},
        index=weekly.index,
    )
    pivot = df.pivot_table(index="year", columns="week", values="value", aggfunc="last")
    rows = []
    for ts, year, week in zip(df.index, df["year"], df["week"], strict=True):
        prior = pivot.loc[(pivot.index < year) & (pivot.index >= year - years), week].dropna()
        rows.append(
            {
                "date": ts,
                "avg": prior.mean() if len(prior) else np.nan,
                "min": prior.min() if len(prior) else np.nan,
                "max": prior.max() if len(prior) else np.nan,
                "n_years": len(prior),
            }
        )
    return pd.DataFrame(rows).set_index("date")
