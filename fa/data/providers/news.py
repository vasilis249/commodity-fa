"""Headlines from RSS feeds (Google News search feeds and publisher feeds).

Feeds only carry recent items, so news is a live-only input: it cannot be replayed
historically and must not be used in backtests.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd

from fa.config import ProviderLimits
from fa.data.http import HttpClient

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}+when:7d&hl=en-US&gl=US&ceid=US:en"
COLUMNS = ["published_at", "title", "source", "link", "feed"]


def feed_url(query_or_url: str) -> str:
    if query_or_url.startswith(("http://", "https://")):
        return query_or_url
    return GOOGLE_NEWS.format(q=quote_plus(query_or_url))


def _parse_date(text: str | None) -> pd.Timestamp:
    if not text:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return pd.to_datetime(text, utc=True, errors="coerce")
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def parse_rss(xml_text: str, feed: str) -> pd.DataFrame:
    """RSS 2.0 -> one row per item (UTC `published_at`)."""
    root = ET.fromstring(xml_text)
    rows = []
    for item in root.iter("item"):
        source_el = item.find("source")
        rows.append(
            {
                "published_at": _parse_date(item.findtext("pubDate")),
                "title": (item.findtext("title") or "").strip(),
                "source": source_el.text.strip()
                if source_el is not None and source_el.text
                else "",
                "link": (item.findtext("link") or "").strip(),
                "feed": feed,
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame.sort_values("published_at", ascending=False, ignore_index=True)


class NewsProvider:
    name = "news"

    def __init__(self, limits: ProviderLimits):
        self.http = HttpClient(limits, "news")

    @property
    def requests_made(self) -> int:
        return self.http.requests_made

    def fetch_feed(self, query_or_url: str) -> pd.DataFrame:
        resp = self.http.get(feed_url(query_or_url))
        return parse_rss(resp.text, feed=query_or_url)
