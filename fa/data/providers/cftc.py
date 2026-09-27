"""CFTC Commitments of Traders, disaggregated futures-only report (public Socrata API).

No key needed. Positions are as of Tuesday; the release time is applied by the data
service through `config/data.yaml: releases.cftc_cot`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from fa.config import ProviderLimits
from fa.data.http import HttpClient
from fa.data.providers.base import finalize

URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
DATE_FIELD = "report_date_as_yyyy_mm_dd"

# Socrata field -> canonical column (all counts of contracts)
FIELDS = {
    "open_interest_all": "open_interest",
    "prod_merc_positions_long": "prod_long",
    "prod_merc_positions_short": "prod_short",
    "swap_positions_long_all": "swap_long",
    "swap__positions_short_all": "swap_short",
    "m_money_positions_long_all": "mm_long",
    "m_money_positions_short_all": "mm_short",
    "m_money_positions_spread": "mm_spread",
    "other_rept_positions_long": "other_long",
    "other_rept_positions_short": "other_short",
    "nonrept_positions_long_all": "nonrept_long",
    "nonrept_positions_short_all": "nonrept_short",
}


def parse_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            col: pd.to_numeric([r.get(f) for r in rows], errors="coerce")
            for f, col in FIELDS.items()
        },
        index=pd.DatetimeIndex([pd.Timestamp(r[DATE_FIELD][:10]) for r in rows]),
    )
    return finalize(frame)


class CFTCProvider:
    name = "cftc"

    def __init__(self, limits: ProviderLimits):
        self.http = HttpClient(limits, "cftc")

    @property
    def requests_made(self) -> int:
        return self.http.requests_made

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        """`key` is the CFTC contract market code, e.g. '067651' (WTI)."""
        where = (
            f"cftc_contract_market_code='{key}' AND "
            f"{DATE_FIELD} between '{start.isoformat()}T00:00:00' and '{end.isoformat()}T23:59:59'"
        )
        rows = self.http.get(
            URL,
            params={
                "$where": where,
                "$select": ",".join([DATE_FIELD, *FIELDS]),
                "$order": DATE_FIELD,
                "$limit": 50000,
            },
        ).json()
        return parse_rows(rows)
