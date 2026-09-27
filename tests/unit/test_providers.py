"""Adapters parse recorded upstream responses (no network)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fa.data.http import DataUnavailable
from fa.data.providers import cftc, eia, fred, news
from tests.conftest import FIXTURES, load_yahoo_fixture


def test_yahoo_normalize_gives_session_dates() -> None:
    frame = load_yahoo_fixture("CL=F", "2026")
    assert frame.index.tz is None
    assert frame.index.name == "date"
    assert (frame.index == frame.index.normalize()).all()
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert {"open", "high", "low", "close", "adj_close", "volume"} <= set(frame.columns)
    assert frame.loc["2026-09-23", "close"] == pytest.approx(92.16, abs=0.01)


def test_eia_parse_recorded_response() -> None:
    payload = json.loads((FIXTURES / "eia_seriesid_WCESTUS1.json").read_text())
    frame, total = eia.parse_response(payload)
    assert total > len(frame) == 6
    assert frame.index.is_monotonic_increasing
    assert frame.loc["2026-09-18", "value"] == 426398
    assert frame["units"].iloc[0] == "MBBL"


def test_eia_monthly_period_maps_to_month_end() -> None:
    assert eia.parse_period("2026-02") == pd.Timestamp("2026-02-28")
    assert eia.parse_period("2026-09-18") == pd.Timestamp("2026-09-18")


def test_eia_error_payload_raises() -> None:
    with pytest.raises(DataUnavailable):
        eia.parse_response({"error": "invalid api key"})


def test_cftc_parse_recorded_rows() -> None:
    rows = json.loads((FIXTURES / "cftc_disagg_067651.json").read_text())
    frame = cftc.parse_rows(rows)
    assert list(frame.columns) == list(cftc.FIELDS.values())
    last = frame.loc["2026-09-22"]
    assert last["open_interest"] == 1841811
    assert last["mm_long"] == 223190 and last["swap_short"] == 584713


def test_fred_parse_marks_missing_values() -> None:
    payload = json.loads((FIXTURES / "fred_observations_DGS10_synthetic.json").read_text())
    frame, total = fred.parse_observations(payload)
    assert total == 5
    assert np.isnan(frame.loc["2026-09-16", "value"])  # "." in FRED
    assert frame.loc["2026-09-18", "first_published"] == pd.Timestamp("2026-09-21")


def test_fred_requires_key() -> None:
    from fa.config import ProviderLimits

    with pytest.raises(DataUnavailable, match="FRED_API_KEY"):
        fred.FREDProvider(ProviderLimits(min_interval_s=0, max_retries=0), None).fetch(
            "DGS10", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-01").date()
        )


@pytest.mark.parametrize("name", ["rss_google_news.xml", "rss_eia_todayinenergy.xml"])
def test_rss_parse(name: str) -> None:
    frame = news.parse_rss((FIXTURES / name).read_text(), feed=name)
    assert len(frame) >= 4
    assert str(frame["published_at"].dt.tz) == "UTC"
    assert frame["published_at"].is_monotonic_decreasing
    assert frame["title"].str.len().gt(0).all()


def test_google_news_items_carry_source() -> None:
    frame = news.parse_rss((FIXTURES / "rss_google_news.xml").read_text(), feed="q")
    assert frame["source"].str.len().gt(0).all()


def test_feed_url() -> None:
    assert news.feed_url("https://x.org/rss") == "https://x.org/rss"
    assert "q=WTI+crude" in news.feed_url("WTI crude")
