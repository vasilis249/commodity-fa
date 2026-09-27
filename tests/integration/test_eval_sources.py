"""Independent check of the eval answer key against the raw providers (network).

The frozen answers come from our own analytics; these tests re-derive a sample straight
from Yahoo, EIA and CFTC with plain HTTP calls (no fa.data code), so a bug in our data
layer cannot make the key agree with itself. Run: `uv run pytest -m network -k sources`.
"""

from __future__ import annotations

import os

import pytest
import requests

from fa.evals import load_questions

pytestmark = pytest.mark.network
KEY = {q.id: q for q in load_questions()}
UA = {"User-Agent": "Mozilla/5.0 (research; commodity-fa eval check)"}


def _yahoo_close(symbol: str, day: str) -> float:
    import pandas as pd
    import yfinance as yf

    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    h = yf.Ticker(symbol).history(start=day, end=end, auto_adjust=False)
    return float(h["Close"].iloc[0])


def _eia(series: str, period: str) -> float:
    r = requests.get(
        f"https://api.eia.gov/v2/seriesid/{series}",
        params={"api_key": os.environ.get("EIA_API_KEY") or "DEMO_KEY"},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    return float(next(x["value"] for x in rows if x["period"] == period))


def _cftc_mm_net(code: str, report_date: str) -> float:
    r = requests.get(
        "https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
        params={
            "cftc_contract_market_code": code,
            "report_date_as_yyyy_mm_dd": f"{report_date}T00:00:00.000",
        },
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    row = r.json()[0]
    return float(row["m_money_positions_long_all"]) - float(row["m_money_positions_short_all"])


def test_wti_settles_match_yahoo() -> None:
    assert _yahoo_close("CL=F", "2026-06-01") == pytest.approx(
        KEY["price-01"].expect.value, abs=0.01
    )
    assert _yahoo_close("CL=F", "2020-04-20") == pytest.approx(-37.63, abs=0.01)
    assert KEY["price-02"].expect.value == pytest.approx(-37.63)


def test_inventories_match_eia() -> None:
    # latest release by the decision date: week ending Fri 2026-05-22 (released Thu 05-28)
    assert _eia("PET.WCESTUS1.W", "2026-05-22") == KEY["fund-01"].expect.value
    # released Thu 2025-11-13 for the week ending 2025-11-07
    assert _eia("NG.NW2_EPG0_SWO_R48_BCF.W", "2025-11-07") == KEY["fund-03"].expect.value


def test_positioning_matches_cftc() -> None:
    assert KEY["cot-04"].expect.value == "2026-05-19"
    assert _cftc_mm_net("067651", "2026-05-19") == KEY["cot-01"].expect.value
