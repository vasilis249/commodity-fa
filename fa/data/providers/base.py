"""Provider interface. Adapters normalize upstream data; they never cache or clean."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class RangeProvider(Protocol):
    """Time-series source queried by key and date range.

    `fetch` returns a frame indexed by a tz-naive DatetimeIndex named "date"
    (observation or period date), sorted and unique. An empty frame means "no rows in
    that range", which is not an error; transport failures raise DataUnavailable.
    """

    name: str

    @property
    def requests_made(self) -> int: ...

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame: ...


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate (keep the latest row) and name the index."""
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = "date"
    return frame
