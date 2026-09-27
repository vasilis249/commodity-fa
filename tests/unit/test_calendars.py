from __future__ import annotations

from datetime import date

from fa.data.calendars import (
    add_business_days,
    holidays_between,
    is_business_day,
    last_business_day_of_month,
    sessions,
)


def test_cme_holidays_2026() -> None:
    for d in [
        date(2026, 1, 1),
        date(2026, 4, 3),
        date(2026, 6, 19),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    ]:
        assert not is_business_day(d, "cme"), d
    assert is_business_day(date(2026, 10, 12), "cme")  # Columbus Day: CME energy settles


def test_juneteenth_only_from_2022() -> None:
    assert is_business_day(date(2021, 6, 18), "cme")
    assert not is_business_day(date(2022, 6, 20), "cme")  # observed Monday


def test_uk_bank_holidays_2026() -> None:
    for d in [
        date(2026, 4, 3),
        date(2026, 4, 6),
        date(2026, 5, 4),
        date(2026, 5, 25),
        date(2026, 8, 31),
        date(2026, 12, 28),
    ]:
        assert not is_business_day(d, "uk"), d
    assert is_business_day(date(2026, 8, 31), "cme")


def test_business_day_arithmetic() -> None:
    assert add_business_days(date(2026, 9, 25), 1, "cme") == date(2026, 9, 28)
    assert add_business_days(date(2026, 9, 26), 0, "cme") == date(2026, 9, 28)  # roll forward
    assert add_business_days(date(2026, 9, 8), -1, "cme") == date(2026, 9, 4)  # over Labor Day
    assert last_business_day_of_month(2026, 8, "uk") == date(2026, 8, 28)
    assert last_business_day_of_month(2026, 8, "cme") == date(2026, 8, 31)


def test_sessions_and_holiday_listing() -> None:
    s = sessions(date(2026, 9, 1), date(2026, 9, 11), "cme")
    assert len(s) == 8  # 9 weekdays minus Labor Day
    assert holidays_between(date(2026, 9, 1), date(2026, 9, 30), "us_federal") == [date(2026, 9, 7)]


def test_saturday_new_year_not_observed_on_friday() -> None:
    assert is_business_day(date(2021, 12, 31), "cme")
    assert not is_business_day(date(2023, 1, 2), "cme")  # Sunday -> Monday
