"""Contract expiries, roll detection and roll-adjusted returns for continuous futures.

Why: yfinance `=F` series are unadjusted front months. When Yahoo switches to the next
contract, the price jumps by the calendar spread (seen up to ~9% in 2026 data), and
that jump is not a return anyone could earn.

Observed Yahoo behaviour:
- Yahoo reports the volume of the contract it currently tracks. That volume fades as
  traders roll, then jumps when Yahoo switches (10 years of CL/NG/HO/RB/BZ: the largest
  day-over-day jump near expiry fell on the expiry session in ~95% of rolls, and the
  jump was >= 2x in ~90% of them);
- the price switches on that session or the next (Sep 2026, checked against live
  contracts: CL the next day, NG/HO/RB/BZ the same day, sometimes 1-3 sessions early).

Approach:
1. Compute each contract's expiry from the exchange rule (known in advance).
2. In the window [expiry - window_before, expiry + window_after] sessions, the session
   with the largest day-over-day volume jump (>= `min_volume_ratio`) marks the switch.
3. Mask that session's return and the next `mask_after` ones. With no clear jump
   (e.g. TTF, whose volume is unreliable), mask the whole window.
4. Log returns are also masked on non-positive or missing prices (WTI, April 2020).
   Masked returns count as 0 when the adjusted close is rebuilt, anchored at the
   latest close. That drops the true move on masked days, and the output flags it.

Look-ahead note: step 2 may use volume up to `window_after` sessions after a masked
day. It only locates the vendor's splice point (not market information), but features
computed inside an open roll window must treat that window's returns as not final.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fa.config import RollRule, RollSettings
from fa.data.calendars import add_business_days, last_business_day_of_month

MONTH_CODES = "FGHJKMNQUVXZ"


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def expiry_date(rule: RollRule, year: int, month: int, calendar: str) -> date:
    """Last trading day of the contract for delivery month (year, month)."""
    if rule is RollRule.NYMEX_CL:
        # 3 business days before the 25th of the prior month; if the 25th is not a
        # business day, 3 business days before the business day preceding it.
        py, pm = _shift_month(year, month, -1)
        anchor = add_business_days(date(py, pm, 25), 0, calendar)
        if anchor != date(py, pm, 25):
            anchor = add_business_days(date(py, pm, 25), -1, calendar)
        return add_business_days(anchor, -3, calendar)
    if rule is RollRule.NYMEX_NG:
        # 3 business days prior to the first calendar day of the delivery month.
        return add_business_days(date(year, month, 1), -3, calendar)
    if rule is RollRule.NYMEX_LAST_BD_PRIOR_MONTH:
        return last_business_day_of_month(*_shift_month(year, month, -1), calendar)
    if rule is RollRule.ICE_BRENT:
        # Last business day of the second month preceding the delivery month.
        return last_business_day_of_month(*_shift_month(year, month, -2), calendar)
    if rule is RollRule.ICE_ENDEX_TTF:
        # 2 business days prior to the first calendar day of the delivery month.
        return add_business_days(date(year, month, 1), -2, calendar)
    raise ValueError(f"{rule} has no expiry schedule")


def contract_code(root: str, year: int, month: int) -> str:
    """e.g. contract_code('CL', 2026, 12) -> 'CLZ26'."""
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}"


def expiries(rule: RollRule, start: date, end: date, calendar: str) -> pd.Series:
    """Expiry dates falling in [start, end], as a Series of 'YYYY-MM' delivery labels."""
    out: dict[pd.Timestamp, str] = {}
    y, m = _shift_month(start.year, start.month, -1)
    while (y, m) <= _shift_month(end.year, end.month, 3):
        e = expiry_date(rule, y, m, calendar)
        if start <= e <= end:
            out[pd.Timestamp(e)] = f"{y}-{m:02d}"
        y, m = _shift_month(y, m, 1)
    return pd.Series(out, name="delivery", dtype="object").sort_index()


@dataclass(frozen=True)
class RollEvent:
    expiry: date
    delivery: str
    switch_session: date | None  # session with the volume jump; None if not detected
    masked: tuple[date, ...]
    method: str  # "volume" | "calendar_window"
    provisional: bool = False  # window extends past the last bar: may still change


def detect_rolls(
    prices: pd.DataFrame, expiry_dates: pd.Series, settings: RollSettings, calendar: str
) -> list[RollEvent]:
    """Find Yahoo's contract switches around each expiry (see module docstring).

    Windows are built on the exchange calendar, so an expiry after the last bar still
    masks the sessions already inside its window (the live, still-open roll).
    """
    idx = prices.index
    if len(idx) < 2:
        return []
    vol = prices["volume"].astype(float)
    jump = (vol / vol.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    events: list[RollEvent] = []
    for exp_ts, delivery in expiry_dates.items():
        expiry = exp_ts.date()
        w_start = pd.Timestamp(add_business_days(expiry, -settings.window_before, calendar))
        w_end = pd.Timestamp(add_business_days(expiry, settings.window_after, calendar))
        window = idx[(idx >= w_start) & (idx <= w_end) & (idx > idx[0])]
        if len(window) == 0:
            continue
        provisional = w_end > idx[-1]
        in_window = jump.loc[window].dropna()
        best = in_window.idxmax() if not in_window.empty else None
        if best is not None and in_window[best] >= settings.min_volume_ratio:
            p = idx.get_loc(best)
            masked = idx[p : p + settings.mask_after + 1]
            events.append(
                RollEvent(
                    expiry=expiry,
                    delivery=str(delivery),
                    switch_session=best.date(),
                    masked=tuple(d.date() for d in masked),
                    method="volume",
                    provisional=provisional,
                )
            )
        else:
            events.append(
                RollEvent(
                    expiry=expiry,
                    delivery=str(delivery),
                    switch_session=None,
                    masked=tuple(d.date() for d in window),
                    method="calendar_window",
                    provisional=provisional,
                )
            )
    return events


def adjusted_returns(prices: pd.DataFrame, roll_events: list[RollEvent]) -> pd.DataFrame:
    """Log close-to-close returns with roll/bad-price days masked, plus an adjusted close.

    Columns added: `ret` (log return, NaN when masked), `mask_reason`
    ("roll" | "nonpositive_price" | "missing" | ""), `close_adj` (continuous close
    anchored at the latest close; masked returns count as zero).
    """
    out = prices.copy()
    close = out["close"].astype(float)
    prev = close.shift(1)

    reason = pd.Series("", index=out.index, dtype="object")
    reason[close.isna() | prev.isna()] = "missing"
    reason[(close <= 0) | (prev <= 0)] = "nonpositive_price"
    rolled = {pd.Timestamp(d) for ev in roll_events for d in ev.masked}
    reason[out.index.isin(list(rolled)) & (reason == "")] = "roll"
    reason.iloc[0] = "missing"  # no prior close

    valid = reason == ""
    ret = pd.Series(np.nan, index=out.index)
    ret[valid] = np.log(close[valid] / prev[valid])

    out["ret"] = ret
    out["mask_reason"] = reason
    out["close_adj"] = _anchor_to_last(close, ret)
    return out


def _anchor_to_last(close: pd.Series, ret: pd.Series) -> pd.Series:
    """Continuous close: last positive close, walked backwards through valid returns."""
    positive = close[close > 0]
    if positive.empty:
        return pd.Series(np.nan, index=close.index)
    r = ret.fillna(0.0)
    # sum of returns strictly after each date
    after = r[::-1].cumsum()[::-1].shift(-1).fillna(0.0)
    anchor_date = positive.index[-1]
    anchor = float(positive.iloc[-1])
    level = anchor * np.exp(-(after - after.loc[anchor_date]))
    return level.rename("close_adj")


def roll_adjust(
    prices: pd.DataFrame, rule: RollRule, calendar: str, settings: RollSettings
) -> tuple[pd.DataFrame, list[RollEvent]]:
    """Detect rolls (futures only) and return the adjusted frame plus the roll events."""
    if prices.empty:
        return adjusted_returns(prices, []), []
    events: list[RollEvent] = []
    if rule is not RollRule.NONE:
        # include expiries just after the last bar: their windows may already be open
        horizon = add_business_days(prices.index[-1].date(), settings.window_before, calendar)
        exp = expiries(rule, prices.index[0].date(), horizon, calendar)
        events = detect_rolls(prices, exp, settings, calendar)
    return adjusted_returns(prices, events), events
