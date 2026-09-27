"""Data service: the only entry point upper layers use to get data.

Combines providers, the disk cache, point-in-time availability and quality checks.

Cache policy for ranged series:
- A cache entry covers [coverage_start, last network fetch].
- A request that starts before coverage_start fetches just the missing head.
- A stale entry (older than the TTL in config) fetches just the tail, from a few days
  before the last cached row (so late corrections overwrite).
- `offline=True` never touches the network and fails if nothing is cached.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

import pandas as pd

from fa.config import AppConfig, Instrument, RollRule, Secrets, load_secrets
from fa.data.cache import CacheMeta, DiskCache
from fa.data.calendars import calendar_for_exchange
from fa.data.http import DataUnavailable
from fa.data.pit import next_day_availability, release_timestamp
from fa.data.providers.base import RangeProvider, finalize
from fa.data.providers.cftc import CFTCProvider
from fa.data.providers.eia import EIAProvider
from fa.data.providers.fred import FREDProvider
from fa.data.providers.news import NewsProvider
from fa.data.providers.yahoo import YahooProvider
from fa.data.quality import QualityReport, check_prices
from fa.data.rolls import RollEvent, contract_code, expiry_date, roll_adjust

log = logging.getLogger(__name__)

TAIL_OVERLAP_DAYS = 7
NEWS_LOOKBACK_DAYS = 7
Source = Literal["cache", "network", "cache+network"]


@dataclass(frozen=True)
class FetchInfo:
    provider: str
    key: str
    source: Source
    requests: int
    rows: int
    fetched_at: datetime
    stale: bool = False


@dataclass(frozen=True)
class PriceData:
    instrument: Instrument
    frame: pd.DataFrame  # OHLCV + ret, mask_reason, close_adj
    rolls: list[RollEvent]
    quality: QualityReport
    info: FetchInfo


def adhoc_instrument(symbol: str) -> Instrument:
    """Instrument for a ticker outside the configured universe (no roll handling)."""
    return Instrument(
        symbol=symbol,
        name=symbol,
        asset_class="equity",
        sector="unknown",
        exchange="NYSE Arca",
        currency="USD",
        unit="USD/share",
        settle_time=time(16, 0),
        timezone="America/New_York",
        roll_rule=RollRule.NONE,
    )


class DataService:
    def __init__(
        self,
        cfg: AppConfig,
        secrets: Secrets | None = None,
        cache: DiskCache | None = None,
        providers: dict[str, RangeProvider] | None = None,
        news: NewsProvider | None = None,
        offline: bool = False,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.cfg = cfg
        self.secrets = secrets or load_secrets()
        cache_root = cfg.data.cache_dir
        if not cache_root.is_absolute():
            cache_root = cfg.config_dir.parent / cache_root
        self.cache = cache or DiskCache(cache_root)
        self._providers: dict[str, RangeProvider] = dict(providers or {})
        self._news = news
        self.offline = offline
        self.now = now

    # --- providers (created lazily so cache-only runs never touch the network) ------

    def _provider(self, name: str) -> RangeProvider:
        if name not in self._providers:
            limits = self.cfg.data.providers[name]
            key = self.secrets
            factories: dict[str, Callable[[], RangeProvider]] = {
                "yahoo": lambda: YahooProvider(limits, self.cache.root / "_yfinance_tz"),
                "eia": lambda: EIAProvider(limits, _reveal(key.eia_api_key)),
                "cftc": lambda: CFTCProvider(limits),
                "fred": lambda: FREDProvider(limits, _reveal(key.fred_api_key)),
            }
            self._providers[name] = factories[name]()
        return self._providers[name]

    def _news_provider(self) -> NewsProvider:
        if self._news is None:
            self._news = NewsProvider(self.cfg.data.providers["news_rss"])
        return self._news

    # --- generic ranged fetch with cache ---------------------------------------------

    def ranged(
        self,
        provider: str,
        key: str,
        start: date,
        end: date | None,
        ttl_hours: float,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, FetchInfo]:
        now = self.now()
        end = end or now.date()
        entry = None if refresh else self.cache.load(provider, key)

        if self.offline:
            if entry is None:
                raise DataUnavailable(f"offline and no cached {provider}:{key}")
            stale = entry.meta.age_hours(now) > ttl_hours
            info = FetchInfo(
                provider, key, "cache", 0, len(entry.frame), entry.meta.fetched_at, stale
            )
            return entry.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)], info

        parts: list[pd.DataFrame] = []
        requests_before = self._requests(provider)
        fetched_at = entry.meta.fetched_at if entry else now
        coverage_start = entry.meta.coverage_start if entry else None

        if entry is None:
            parts.append(self._provider(provider).fetch(key, start, end))
            coverage_start = start
        else:
            parts.append(entry.frame)
            if coverage_start is None or start < coverage_start:
                head_end = (coverage_start or end) - timedelta(days=1)
                parts.append(self._provider(provider).fetch(key, start, head_end))
                coverage_start = start
            last_row = entry.frame.index.max() if not entry.frame.empty else None
            stale = entry.meta.age_hours(now) > ttl_hours
            if stale and (last_row is None or pd.Timestamp(end) >= last_row):
                tail_start = (
                    last_row.date() - timedelta(days=TAIL_OVERLAP_DAYS)
                    if last_row is not None
                    else coverage_start
                )
                parts.append(self._provider(provider).fetch(key, tail_start, end))
                fetched_at = now

        requests = self._requests(provider) - requests_before
        non_empty = [p for p in parts if not p.empty]
        frame = finalize(pd.concat(non_empty)) if non_empty else parts[0]
        if frame.empty:
            raise DataUnavailable(f"{provider}:{key} returned no data for {start}..{end}")

        if requests:
            self.cache.save(
                frame,
                CacheMeta(
                    provider=provider,
                    key=key,
                    fetched_at=fetched_at,
                    coverage_start=coverage_start,
                    rows=len(frame),
                ),
            )
        source: Source = "network" if entry is None else ("cache+network" if requests else "cache")
        info = FetchInfo(provider, key, source, requests, len(frame), fetched_at)
        return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)], info

    def _requests(self, provider: str) -> int:
        p = self._providers.get(provider)
        return p.requests_made if p is not None else 0

    # --- prices ------------------------------------------------------------------------

    def instrument(self, symbol: str) -> Instrument:
        return self.cfg.data.instrument(symbol) or adhoc_instrument(symbol)

    def prices(
        self,
        symbol: str,
        years: float | None = None,
        refresh: bool = False,
        as_of: date | None = None,
    ) -> PriceData:
        """Daily prices with roll/bad-price masked returns and a quality report.

        With `as_of`, bars after that session are dropped *before* roll detection and
        quality checks, so nothing computed here can see data after `as_of`.
        """
        inst = self.instrument(symbol)
        years = years or self.cfg.data.default_history_years
        today = self.now().date()
        end = min(as_of, today) if as_of else today
        start = end - timedelta(days=round(365.25 * years))
        raw, info = self.ranged(
            "yahoo", symbol, start, today, self.cfg.data.cache_ttl.prices_hours, refresh
        )
        raw = raw.loc[: pd.Timestamp(end)]
        if raw.empty:
            raise DataUnavailable(f"{symbol}: no prices on or before {end}")
        calendar = calendar_for_exchange(inst.exchange)
        adjusted, events = roll_adjust(raw, inst.roll_rule, calendar, self.cfg.data.rolls)
        report = check_prices(raw, inst, self.cfg.data.quality, events, adjusted["ret"], today=end)
        return PriceData(inst, adjusted, events, report, info)

    def curve(
        self, symbol: str, contracts: int | None = None, refresh: bool = False
    ) -> tuple[pd.DataFrame, list[FetchInfo]]:
        """Latest settlement of the next `contracts` listed contract months.

        Columns: contract, delivery, expiry, settle, settle_date. Only the live curve is
        available (Yahoo drops expired contracts), so there is no historical as-of.
        """
        inst = self.instrument(symbol)
        if not (inst.contract_root and inst.contract_suffix) or inst.roll_rule is RollRule.NONE:
            raise DataUnavailable(f"{symbol}: no single-contract symbols configured")
        n = contracts or self.cfg.data.curve_contracts
        today = self.now().date()
        calendar = calendar_for_exchange(inst.exchange)
        rows: list[dict[str, object]] = []
        infos: list[FetchInfo] = []
        y, m = today.year, today.month
        while len(rows) < n and (y - today.year) * 12 + (m - today.month) < n + 24:
            exp = expiry_date(inst.roll_rule, y, m, calendar)
            if exp >= today:
                code = contract_code(inst.contract_root, y, m) + inst.contract_suffix
                try:
                    frame, info = self.ranged(
                        "yahoo",
                        code,
                        today - timedelta(days=30),
                        today,
                        self.cfg.data.cache_ttl.prices_hours,
                        refresh,
                    )
                except DataUnavailable as exc:
                    log.warning("curve: %s unavailable (%s)", code, exc)
                else:
                    last = frame["close"].dropna()
                    if not last.empty:
                        rows.append(
                            {
                                "contract": code.removesuffix(inst.contract_suffix),
                                "delivery": f"{y}-{m:02d}",
                                "expiry": exp,
                                "settle": float(last.iloc[-1]),
                                "settle_date": last.index[-1].date(),
                            }
                        )
                        infos.append(info)
            m += 1
            if m == 13:
                y, m = y + 1, 1
        if len(rows) < 2:
            raise DataUnavailable(f"{symbol}: fewer than 2 listed contracts found")
        return pd.DataFrame(rows), infos

    # --- fundamentals / positioning / macro ---------------------------------------------

    def eia(
        self, series_id: str, start: date | None = None, refresh: bool = False
    ) -> tuple[pd.DataFrame, FetchInfo]:
        """EIA series with `available_at` (UTC) from the release schedule."""
        schedule_name = self.cfg.data.eia_release_by_prefix.get(series_id.split(".", 1)[0])
        if schedule_name is None:
            raise ValueError(f"no release schedule configured for EIA series {series_id}")
        schedule = self.cfg.data.releases[schedule_name]
        frame, info = self.ranged(
            "eia",
            series_id,
            start or self._default_start(),
            None,
            self.cfg.data.cache_ttl.eia_hours,
            refresh,
        )
        frame = frame.copy()
        frame["available_at"] = [release_timestamp(d.date(), schedule) for d in frame.index]
        return frame, info

    def cot(
        self, symbol: str, start: date | None = None, refresh: bool = False
    ) -> tuple[pd.DataFrame, FetchInfo]:
        """CFTC disaggregated positions for an instrument, with `available_at`."""
        inst = self.instrument(symbol)
        if not inst.cot_market_code:
            raise DataUnavailable(f"{symbol} has no CFTC market code in config")
        schedule = self.cfg.data.releases["cftc_cot"]
        frame, info = self.ranged(
            "cftc",
            inst.cot_market_code,
            start or self._default_start(),
            None,
            self.cfg.data.cache_ttl.cot_hours,
            refresh,
        )
        frame = frame.copy()
        frame["available_at"] = [release_timestamp(d.date(), schedule) for d in frame.index]
        return frame, info

    def fred(
        self, series_id: str, start: date | None = None, refresh: bool = False
    ) -> tuple[pd.DataFrame, FetchInfo]:
        """FRED first-release values with `available_at` (next day after publication)."""
        frame, info = self.ranged(
            "fred",
            series_id,
            start or self._default_start(),
            None,
            self.cfg.data.cache_ttl.fred_hours,
            refresh,
        )
        frame = frame.copy()
        frame["available_at"] = next_day_availability(
            frame["first_published"], self.cfg.data.fred_availability_lag_days, "America/New_York"
        ).to_numpy()
        return frame, info

    def _default_start(self) -> date:
        return self.now().date() - timedelta(
            days=round(365.25 * self.cfg.data.default_history_years)
        )

    # --- news (snapshot cache, live-only) -----------------------------------------------

    def news(self, symbol: str, refresh: bool = False) -> tuple[pd.DataFrame, list[FetchInfo]]:
        inst = self.instrument(symbol)
        feeds = [*inst.news_queries, *self.cfg.data.news_feeds] or [symbol]
        ttl = self.cfg.data.cache_ttl.news_hours
        frames, infos = [], []
        for feed in feeds:
            try:
                frame, info = self._news_snapshot(feed, ttl, refresh)
            except DataUnavailable as exc:
                log.warning("news feed skipped: %s", exc)
                continue
            frames.append(frame)
            infos.append(info)
        if not frames:
            raise DataUnavailable(f"no news feed returned data for {symbol}")
        allnews = pd.concat(frames, ignore_index=True)
        cutoff = self.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
        allnews = allnews[allnews["published_at"] >= cutoff]
        allnews = allnews.drop_duplicates(subset="title").sort_values(
            "published_at", ascending=False, ignore_index=True
        )
        return allnews, infos

    def _news_snapshot(
        self, feed: str, ttl: float, refresh: bool
    ) -> tuple[pd.DataFrame, FetchInfo]:
        now = self.now()
        entry = None if refresh else self.cache.load("news", feed)
        if entry is not None and (self.offline or entry.meta.age_hours(now) <= ttl):
            stale = entry.meta.age_hours(now) > ttl
            return entry.frame, FetchInfo(
                "news", feed, "cache", 0, len(entry.frame), entry.meta.fetched_at, stale
            )
        if self.offline:
            raise DataUnavailable(f"offline and no cached news for {feed!r}")
        provider = self._news_provider()
        before = provider.requests_made
        frame = provider.fetch_feed(feed)
        self.cache.save(
            frame, CacheMeta(provider="news", key=feed, fetched_at=now, rows=len(frame))
        )
        return frame, FetchInfo(
            "news", feed, "network", provider.requests_made - before, len(frame), now
        )


def _reveal(secret: object) -> str | None:
    return secret.get_secret_value() if secret is not None else None  # type: ignore[attr-defined]
