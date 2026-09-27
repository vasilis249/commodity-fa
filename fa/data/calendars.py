"""Exchange and publisher business-day calendars.

Holiday rules are approximations of the official calendars, good enough for contract
expiry dates, release-lag timing and gap detection. One-off closures (e.g. national
days of mourning) are not modelled; callers must tolerate a day of slack.
"""

from __future__ import annotations

from datetime import date
from functools import cache

import pandas as pd
from pandas.tseries.holiday import (
    MO,
    AbstractHolidayCalendar,
    EasterMonday,
    GoodFriday,
    Holiday,
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    next_monday,
    next_monday_or_tuesday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class CMEHolidayCalendar(AbstractHolidayCalendar):
    """NYMEX/CME energy settlement holidays (no settlement price on these days)."""

    rules = [  # noqa: RUF012 - pandas API expects a class attribute list
        # a Saturday New Year's Day is not observed on the Friday (CME, NYSE)
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


class UKHolidayCalendar(AbstractHolidayCalendar):
    """England & Wales bank holidays (ICE Futures Europe / ICE Endex business days)."""

    rules = [  # noqa: RUF012
        Holiday("NewYearsDay", month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday("EarlyMay", month=5, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("SpringBank", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("SummerBank", month=8, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("Christmas", month=12, day=25, observance=next_monday),
        Holiday("BoxingDay", month=12, day=26, observance=next_monday_or_tuesday),
    ]


CALENDARS: dict[str, type[AbstractHolidayCalendar]] = {
    "cme": CMEHolidayCalendar,
    "uk": UKHolidayCalendar,
    "us_federal": USFederalHolidayCalendar,  # EIA / CFTC publication schedules
}

EXCHANGE_CALENDAR: dict[str, str] = {
    "NYMEX": "cme",
    "ICE": "uk",
    "ICE Endex": "uk",
    "NYSE Arca": "cme",  # close enough for gap checks; ETF holidays match CME's here
}


@cache
def business_day(calendar: str) -> CustomBusinessDay:
    """A business-day offset that skips weekends and the calendar's holidays."""
    return CustomBusinessDay(calendar=CALENDARS[calendar]())


@cache
def _holidays(calendar: str) -> frozenset[date]:
    days = CALENDARS[calendar]().holidays(start="1990-01-01", end="2040-12-31")
    return frozenset(d.date() for d in days)


def is_business_day(d: date, calendar: str) -> bool:
    return d.weekday() < 5 and d not in _holidays(calendar)


def holidays_between(start: date, end: date, calendar: str) -> list[date]:
    """Weekday holidays in [start, end], inclusive."""
    return sorted(h for h in _holidays(calendar) if start <= h <= end and h.weekday() < 5)


def sessions(start: date, end: date, calendar: str) -> pd.DatetimeIndex:
    """Expected trading sessions in [start, end]."""
    return pd.date_range(start, end, freq=business_day(calendar))


def add_business_days(d: date, n: int, calendar: str) -> date:
    """Move n business days from d (n may be negative). n=0 rolls forward to a business day."""
    ts = pd.Timestamp(d)
    offset = business_day(calendar)
    if n == 0:
        return offset.rollforward(ts).date()
    return (ts + n * offset).date()


def last_business_day_of_month(year: int, month: int, calendar: str) -> date:
    end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return business_day(calendar).rollback(end).date()


def calendar_for_exchange(exchange: str) -> str:
    return EXCHANGE_CALENDAR.get(exchange, "cme")
