"""EIA API v2 (weekly petroleum and natural-gas storage data).

Uses the `/v2/seriesid/<legacy id>` route so config can keep the familiar legacy ids
(e.g. PET.WCESTUS1.W). Without EIA_API_KEY it falls back to api.data.gov's DEMO_KEY,
which works but is heavily rate-limited.

Limitation: the API serves the latest revision of each value, not the first print.
Weekly petroleum revisions are rare and small; this is noted for the leakage auditor.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from fa.config import ProviderLimits
from fa.data.http import DataUnavailable, HttpClient
from fa.data.providers.base import finalize

log = logging.getLogger(__name__)

BASE_URL = "https://api.eia.gov/v2/seriesid/"
PAGE = 5000


def parse_period(period: str) -> pd.Timestamp:
    """'2026-09-18' (weekly/daily) or '2026-09' (monthly, mapped to month end)."""
    if len(period) == 7:
        return pd.Timestamp(period + "-01") + pd.offsets.MonthEnd(0)
    return pd.Timestamp(period)


def parse_response(payload: dict) -> tuple[pd.DataFrame, int]:
    """EIA v2 JSON -> (frame with value/units, total row count on the server)."""
    if "response" not in payload:
        raise DataUnavailable(f"EIA error: {payload.get('error') or payload}")
    resp = payload["response"]
    rows = resp.get("data", [])
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric([r.get("value") for r in rows], errors="coerce"),
            "units": [r.get("units") for r in rows],
        },
        index=pd.DatetimeIndex([parse_period(r["period"]) for r in rows]),
    )
    return finalize(frame), int(resp.get("total", len(rows)))


class EIAProvider:
    name = "eia"

    def __init__(self, limits: ProviderLimits, api_key: str | None):
        if not api_key:
            log.warning("EIA_API_KEY not set; using DEMO_KEY (rate-limited)")
        self.api_key = api_key or "DEMO_KEY"
        self.http = HttpClient(limits, "eia")

    @property
    def requests_made(self) -> int:
        return self.http.requests_made

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            payload = self.http.get(
                BASE_URL + key,
                params={
                    "api_key": self.api_key,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "offset": offset,
                    "length": PAGE,
                },
            ).json()
            frame, total = parse_response(payload)
            frames.append(frame)
            offset += PAGE
            if offset >= total or frame.empty:
                break
        return finalize(pd.concat(frames)) if frames else finalize(pd.DataFrame())
