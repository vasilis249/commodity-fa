"""FRED / ALFRED observations, first-release values only.

`output_type=4` returns each observation as initially published, with the date it was
first published (`realtime_start`). That avoids revision leakage and gives the data
service what it needs for `available_at`. Requires FRED_API_KEY (free).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from fa.config import ProviderLimits
from fa.data.http import DataUnavailable, HttpClient
from fa.data.providers.base import finalize

URL = "https://api.stlouisfed.org/fred/series/observations"
PAGE = 100000


def parse_observations(payload: dict) -> tuple[pd.DataFrame, int]:
    if "observations" not in payload:
        raise DataUnavailable(f"FRED error: {payload.get('error_message') or payload}")
    obs = payload["observations"]
    frame = pd.DataFrame(
        {
            # FRED uses "." for missing values
            "value": pd.to_numeric([o["value"] for o in obs], errors="coerce"),
            "first_published": pd.to_datetime([o["realtime_start"] for o in obs]),
        },
        index=pd.DatetimeIndex([pd.Timestamp(o["date"]) for o in obs]),
    )
    return finalize(frame), int(payload.get("count", len(obs)))


class FREDProvider:
    name = "fred"

    def __init__(self, limits: ProviderLimits, api_key: str | None):
        self.api_key = api_key
        self.http = HttpClient(limits, "fred")

    @property
    def requests_made(self) -> int:
        return self.http.requests_made

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        if not self.api_key:
            raise DataUnavailable("FRED_API_KEY is not set (free key: see README)")
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            payload = self.http.get(
                URL,
                params={
                    "series_id": key,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                    "realtime_start": "1776-07-04",
                    "realtime_end": "9999-12-31",
                    "output_type": 4,
                    "offset": offset,
                    "limit": PAGE,
                },
            ).json()
            frame, total = parse_observations(payload)
            frames.append(frame)
            offset += PAGE
            if offset >= total or frame.empty:
                break
        return finalize(pd.concat(frames))
