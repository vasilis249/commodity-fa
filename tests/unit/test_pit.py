"""Point-in-time tests. The leak test proves a period-date join would see the future."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from fa.config import AppConfig
from fa.data.pit import (
    next_day_availability,
    pit_join,
    release_date,
    release_timestamp,
)


@pytest.mark.parametrize(
    ("schedule", "period_end", "expected"),
    [
        ("eia_petroleum_weekly", date(2026, 9, 18), date(2026, 9, 23)),  # normal Wednesday
        ("eia_petroleum_weekly", date(2026, 9, 4), date(2026, 9, 10)),  # Labor Day -> Thursday
        ("eia_gas_storage_weekly", date(2026, 9, 18), date(2026, 9, 24)),  # normal Thursday
        ("eia_gas_storage_weekly", date(2026, 11, 20), date(2026, 11, 27)),  # Thanksgiving
        ("cftc_cot", date(2026, 9, 15), date(2026, 9, 18)),  # normal Friday
        ("cftc_cot", date(2026, 9, 8), date(2026, 9, 14)),  # Labor Day week -> Monday
        ("cftc_cot", date(2026, 12, 22), date(2026, 12, 28)),  # Christmas Friday -> Monday
    ],
)
def test_release_dates(cfg: AppConfig, schedule: str, period_end: date, expected: date) -> None:
    assert release_date(period_end, cfg.data.releases[schedule]) == expected


def test_release_timestamp_handles_dst(cfg: AppConfig) -> None:
    wpsr = cfg.data.releases["eia_petroleum_weekly"]
    assert release_timestamp(date(2026, 9, 18), wpsr) == pd.Timestamp("2026-09-23 14:30", tz="UTC")
    assert release_timestamp(date(2026, 12, 4), wpsr) == pd.Timestamp("2026-12-09 15:30", tz="UTC")


def _weekly_inventory(cfg: AppConfig) -> pd.DataFrame:
    wpsr = cfg.data.releases["eia_petroleum_weekly"]
    periods = pd.to_datetime(["2026-09-04", "2026-09-11", "2026-09-18"])
    frame = pd.DataFrame({"value": [100.0, 200.0, 300.0]}, index=periods)
    frame["available_at"] = [release_timestamp(d.date(), wpsr) for d in periods]
    return frame


def test_pit_join_uses_release_time_not_period_date(cfg: AppConfig) -> None:
    series = _weekly_inventory(cfg)
    sessions = pd.bdate_range("2026-09-18", "2026-09-25")
    joined = pit_join(sessions, series, time(14, 30), "America/New_York")
    # Fri 18th .. Tue 22nd: the 18 Sep week is not public yet (released Wed 23rd 10:30 ET)
    assert joined.loc["2026-09-18", "value"] == 200.0
    assert joined.loc["2026-09-22", "value"] == 200.0
    # Wed 23rd: released 10:30 ET, before the 14:30 ET settlement
    assert joined.loc["2026-09-23", "value"] == 300.0
    assert joined.loc["2026-09-23", "asof_period"] == pd.Timestamp("2026-09-18")


def test_naive_period_date_join_would_leak(cfg: AppConfig) -> None:
    """The leak this module prevents: joining on period date shows data 3 sessions early."""
    series = _weekly_inventory(cfg)
    sessions = pd.bdate_range("2026-09-18", "2026-09-25")
    naive = series["value"].reindex(sessions, method="ffill")
    safe = pit_join(sessions, series, time(14, 30), "America/New_York")["value"]
    leaked = sessions[naive.to_numpy() != safe.to_numpy()]
    assert list(leaked.date) == [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22)]


def test_cot_released_after_settlement_is_used_next_session(cfg: AppConfig) -> None:
    cot = cfg.data.releases["cftc_cot"]
    frame = pd.DataFrame(
        {"mm_long": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-08", "2026-09-15"])
    )
    frame["available_at"] = [release_timestamp(d.date(), cot) for d in frame.index]
    joined = pit_join(
        pd.bdate_range("2026-09-18", "2026-09-21"), frame, time(14, 30), "America/New_York"
    )
    assert joined.loc["2026-09-18", "mm_long"] == 1.0  # released 15:30 ET, after the 14:30 settle
    assert joined.loc["2026-09-21", "mm_long"] == 2.0


def test_pit_join_requires_available_at() -> None:
    with pytest.raises(ValueError, match="available_at"):
        pit_join(
            pd.bdate_range("2026-01-05", periods=2), pd.DataFrame({"v": [1]}), time(14, 30), "UTC"
        )


def test_next_day_availability() -> None:
    first = pd.Series(pd.to_datetime(["2026-09-15", "2026-11-02"]))
    out = next_day_availability(first, 1, "America/New_York")
    assert out.iloc[0] == pd.Timestamp("2026-09-16 04:00", tz="UTC")  # EDT
    assert out.iloc[1] == pd.Timestamp("2026-11-03 05:00", tz="UTC")  # EST
