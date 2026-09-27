"""Point-in-time rules: when each value became public, and as-of joins on that time.

Every non-price series is stored with an `available_at` column (UTC). Features for the
session on date t may only use rows with `available_at <= settlement time of t`. Joining
on the observation/period date instead is the classic leak this module exists to stop.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from fa.config import ReleaseOverride, ReleaseSchedule
from fa.data.calendars import add_business_days, holidays_between

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def release_date(period_end: date, schedule: ReleaseSchedule) -> date:
    """Date a weekly report for `period_end` is published (conservative holiday rule)."""
    target = WEEKDAYS.index(schedule.weekday)
    days_ahead = (target - period_end.weekday()) % 7 or 7  # strictly after period end
    nominal = period_end + timedelta(days=days_ahead)
    week_monday = nominal - timedelta(days=nominal.weekday())
    n_holidays = len(holidays_between(week_monday, nominal, schedule.calendar))
    if n_holidays == 0:
        return nominal
    return add_business_days(nominal, n_holidays, schedule.calendar)


def release_timestamp(period_end: date, schedule: ReleaseSchedule) -> pd.Timestamp:
    """UTC timestamp at which the report covering `period_end` became public."""
    d = release_date(period_end, schedule)
    local = datetime.combine(d, schedule.release_time, tzinfo=ZoneInfo(schedule.tz))
    return pd.Timestamp(local).tz_convert("UTC")


def apply_overrides(
    frame: pd.DataFrame, release: str, schedule: ReleaseSchedule, overrides: list[ReleaseOverride]
) -> pd.DataFrame:
    """Push `available_at` later for periods covered by a release override."""
    out = frame.copy()
    for ov in overrides:
        if ov.release != release:
            continue
        hit = (out.index >= pd.Timestamp(ov.period_start)) & (
            out.index <= pd.Timestamp(ov.period_end)
        )
        if hit.any():
            late = pd.Timestamp(
                datetime.combine(
                    ov.available_from, schedule.release_time, tzinfo=ZoneInfo(schedule.tz)
                )
            ).tz_convert("UTC")
            out.loc[hit, "available_at"] = out.loc[hit, "available_at"].where(
                out.loc[hit, "available_at"] >= late, late
            )
    return out


def next_day_availability(first_published: pd.Series, lag_days: int, tz: str) -> pd.Series:
    """`available_at` for values whose publication time of day is unknown (e.g. FRED).

    Usable from 00:00 local time `lag_days` after the first-publication date.
    """
    local = pd.to_datetime(first_published) + pd.Timedelta(days=lag_days)
    return local.dt.tz_localize(ZoneInfo(tz)).dt.tz_convert("UTC")


def decision_times(sessions: pd.DatetimeIndex, settle_time: time, tz: str) -> pd.DatetimeIndex:
    """UTC decision timestamp for each session date: its settlement time."""
    local = [datetime.combine(d.date(), settle_time, tzinfo=ZoneInfo(tz)) for d in sessions]
    return pd.DatetimeIndex(pd.to_datetime(local, utc=True), name="decision_time")


def pit_join(
    sessions: pd.DatetimeIndex,
    series: pd.DataFrame,
    settle_time: time,
    tz: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Align `series` to `sessions` using only rows public by each session's settlement.

    `series` must have a DatetimeIndex (period/observation date) and an `available_at`
    UTC column. Returns one row per session with the chosen columns, plus
    `asof_period` (the period of the row used) and `asof_available_at`.
    """
    if "available_at" not in series.columns:
        raise ValueError("series needs an 'available_at' column for a point-in-time join")
    cols = columns or [c for c in series.columns if c != "available_at"]
    right = series[[*cols, "available_at"]].copy()
    right["asof_period"] = series.index
    right = right.sort_values("available_at")
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)

    left = pd.DataFrame(
        {"session": sessions, "decision_time": decision_times(sessions, settle_time, tz)}
    ).sort_values("decision_time")
    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.rename(columns={"available_at": "asof_available_at"})
    return merged.set_index("session").drop(columns="decision_time")
