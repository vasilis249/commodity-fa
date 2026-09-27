"""Data-quality checks for daily price frames. Pure functions; nothing is modified.

The report is diagnostic: it tells the user and downstream layers what to distrust.
Cleaning decisions (roll and bad-price masking) live in `fa.data.rolls`.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fa.config import Instrument, QualitySettings, RollRule
from fa.data.calendars import calendar_for_exchange, sessions
from fa.data.rolls import RollEvent

SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 20, 25, 50)
MAX_LISTED_DATES = 20


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class QualityIssue(BaseModel):
    check: str
    severity: Severity
    count: int
    detail: str
    dates: list[date] = []

    @property
    def first(self) -> date | None:
        return self.dates[0] if self.dates else None


class QualityReport(BaseModel):
    symbol: str
    rows: int
    start: date | None
    end: date | None
    issues: list[QualityIssue]

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    def by_check(self, check: str) -> QualityIssue | None:
        return next((i for i in self.issues if i.check == check), None)


def _issue(check: str, sev: Severity, dates: pd.Index | list, detail: str) -> QualityIssue:
    ds = [pd.Timestamp(d).date() for d in dates]
    return QualityIssue(
        check=check, severity=sev, count=len(ds), detail=detail, dates=ds[:MAX_LISTED_DATES]
    )


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Consecutive True runs as (first, last, length)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run: list[pd.Timestamp] = []
    for ts, flag in mask.items():
        if flag:
            run.append(ts)
        elif run:
            out.append((run[0], run[-1], len(run)))
            run = []
    if run:
        out.append((run[0], run[-1], len(run)))
    return out


def check_prices(
    prices: pd.DataFrame,
    instrument: Instrument,
    settings: QualitySettings,
    roll_events: list[RollEvent] | None = None,
    returns: pd.Series | None = None,
    today: date | None = None,
) -> QualityReport:
    """Run every price check. `returns` are the masked log returns from `fa.data.rolls`."""
    today = today or date.today()
    issues: list[QualityIssue] = []
    if prices.empty:
        issues.append(
            QualityIssue(check="empty", severity=Severity.ERROR, count=0, detail="no rows")
        )
        return QualityReport(symbol=instrument.symbol, rows=0, start=None, end=None, issues=issues)

    idx = prices.index
    ohlc = prices[["open", "high", "low", "close"]]
    cal = calendar_for_exchange(instrument.exchange)

    missing = ohlc.isna().any(axis=1)
    if missing.any():
        issues.append(_issue("missing_values", Severity.WARN, idx[missing], "NaN in OHLC"))

    tol = 1e-6 * ohlc.abs().max(axis=1)
    bad = (prices["high"] + tol < ohlc.max(axis=1)) | (prices["low"] - tol > ohlc.min(axis=1))
    if bad.any():
        issues.append(
            _issue("ohlc_inconsistent", Severity.WARN, idx[bad], "high/low outside range")
        )

    nonpos = prices["close"] <= 0
    if nonpos.any():
        issues.append(
            _issue(
                "nonpositive_price",
                Severity.WARN,
                idx[nonpos],
                "close <= 0; log returns around these days are masked",
            )
        )

    if "volume" in prices:
        zero = prices["volume"] == 0
        if zero.any():
            issues.append(_issue("zero_volume", Severity.WARN, idx[zero], "volume == 0"))

    expected = sessions(idx[0].date(), idx[-1].date(), cal)
    absent = pd.Series(~expected.isin(idx), index=expected)
    long_gaps = [r for r in _runs(absent) if r[2] > settings.max_gap_sessions]
    if long_gaps:
        issues.append(
            _issue(
                "gap",
                Severity.WARN,
                [g[0] for g in long_gaps],
                f"{len(long_gaps)} gaps longer than {settings.max_gap_sessions} sessions "
                f"(longest {max(g[2] for g in long_gaps)})",
            )
        )
    if absent.any():
        issues.append(
            _issue(
                "missing_sessions",
                Severity.INFO,
                absent.index[absent],
                f"{int(absent.sum())} expected {cal} sessions without a bar",
            )
        )
    extra = ~idx.isin(expected)
    if extra.any():
        issues.append(
            _issue("unexpected_sessions", Severity.INFO, idx[extra], "bars on weekends/holidays")
        )

    close = prices["close"]
    same = close.eq(close.shift(1))
    long_repeats = [r for r in _runs(same) if r[2] + 1 >= settings.repeated_close_run]
    if long_repeats:
        issues.append(
            _issue(
                "repeated_close",
                Severity.WARN,
                [r[0] for r in long_repeats],
                f"runs of >= {settings.repeated_close_run} identical closes (stale quotes?)",
            )
        )

    # sessions strictly between the last bar and today (today's bar may not exist yet)
    lag = len(sessions(idx[-1].date() + timedelta(days=1), today - timedelta(days=1), cal))
    if lag > settings.stale_sessions:
        issues.append(
            QualityIssue(
                check="stale_data",
                severity=Severity.WARN,
                count=lag,
                detail=f"last bar {idx[-1].date()} is {lag} sessions old",
                dates=[idx[-1].date()],
            )
        )

    if roll_events:
        by_volume = [e for e in roll_events if e.method == "volume"]
        by_window = [e for e in roll_events if e.method != "volume"]
        masked = sum(len(e.masked) for e in roll_events)
        issues.append(
            _issue(
                "rolls",
                Severity.INFO,
                [e.switch_session or e.expiry for e in roll_events],
                f"{len(roll_events)} expiries: {len(by_volume)} switches found by volume, "
                f"{len(by_window)} masked by calendar window; {masked} returns masked",
            )
        )

    if returns is not None:
        r = returns.dropna()
        if len(r) > 30:
            sigma = 1.4826 * float(np.median(np.abs(r - r.median())))
            out = r[np.abs(r) > settings.outlier_mad_multiple * sigma] if sigma > 0 else r.iloc[:0]
            if not out.empty:
                issues.append(
                    _issue(
                        "outlier_return",
                        Severity.INFO,
                        out.index,
                        f"|log return| > {settings.outlier_mad_multiple:g} robust sigmas "
                        f"({sigma:.4f})",
                    )
                )
        if instrument.roll_rule is RollRule.NONE:
            issues += _split_suspects(prices, r)

    return QualityReport(
        symbol=instrument.symbol,
        rows=len(prices),
        start=idx[0].date(),
        end=idx[-1].date(),
        issues=issues,
    )


def _split_suspects(prices: pd.DataFrame, returns: pd.Series) -> list[QualityIssue]:
    """Jumps that look like an unadjusted split (ratio ~ n or 1/n) with no split recorded."""
    recorded = prices.index[prices["splits"].fillna(0) != 0] if "splits" in prices else []
    ratio = np.exp(returns)
    hits = []
    for ts, x in ratio.items():
        if ts in recorded:
            continue
        for n in SPLIT_RATIOS:
            if abs(x - n) / n < 0.02 or abs(x - 1 / n) * n < 0.02:
                hits.append(ts)
                break
    if not hits:
        return []
    return [_issue("split_suspect", Severity.WARN, hits, "price jump matching a split ratio")]
