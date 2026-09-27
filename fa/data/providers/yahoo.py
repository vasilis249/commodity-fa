"""Daily OHLCV from Yahoo Finance via yfinance (prices only; free, unofficial API)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFTickerMissingError, YFTzMissingError

from fa.config import ProviderLimits
from fa.data.http import RateLimiter, TransientError, with_retries
from fa.data.providers.base import finalize

COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance history frame -> canonical price frame (session-date index)."""
    renamed = raw.rename(columns=COLUMNS)
    frame = renamed[[c for c in COLUMNS.values() if c in renamed.columns]]
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    return finalize(frame.astype(float))


class YahooProvider:
    name = "yahoo"

    def __init__(self, limits: ProviderLimits, tz_cache_dir: Path | None = None):
        self.limits = limits
        self.limiter = RateLimiter(limits.min_interval_s)
        self.requests_made = 0
        yf.config.debug.hide_exceptions = False  # raise instead of printing and returning empty
        if tz_cache_dir is not None:
            tz_cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(tz_cache_dir))

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        def once() -> pd.DataFrame:
            self.requests_made += 1
            try:
                return yf.Ticker(key).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )
            except (YFTickerMissingError, YFTzMissingError, YFPricesMissingError):
                return pd.DataFrame()  # permanent: unknown ticker or no rows in range
            except Exception as exc:  # rate limits and unknown upstream errors: retry
                if "404" in str(exc):
                    return pd.DataFrame()  # Yahoo does not know the symbol
                raise TransientError(f"yfinance {key}: {exc}") from exc

        raw = with_retries(once, self.limits, self.limiter, what=f"yahoo {key}")
        if raw.empty:
            return finalize(pd.DataFrame(columns=list(COLUMNS.values()), dtype=float))
        return normalize(raw)
