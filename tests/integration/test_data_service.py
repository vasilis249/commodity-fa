"""DataService cache policy with fake providers (counts network calls, no network)."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from fa.cli import app
from fa.config import AppConfig, Secrets, load_config
from fa.data.cache import DiskCache
from fa.data.http import DataUnavailable
from fa.data.providers.news import parse_rss
from fa.data.service import DataService
from tests.conftest import FIXTURES, load_yahoo_fixture

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


class FakeProvider:
    """Serves slices of a fixed frame and records every call."""

    def __init__(self, name: str, frame: pd.DataFrame):
        self.name = name
        self.frame = frame
        self.calls: list[tuple[str, date, date]] = []

    @property
    def requests_made(self) -> int:
        return len(self.calls)

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((key, start, end))
        return self.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class FakeNews:
    def __init__(self) -> None:
        self.requests_made = 0

    def fetch_feed(self, feed: str) -> pd.DataFrame:
        self.requests_made += 1
        frame = parse_rss((FIXTURES / "rss_google_news.xml").read_text(), feed=feed)
        # make items recent relative to NOW so the 7-day filter keeps them
        frame["published_at"] = NOW - pd.to_timedelta(range(len(frame)), unit="h")
        return frame


def _prices() -> pd.DataFrame:
    return load_yahoo_fixture("CL=F", "2026")


def _svc(cfg: AppConfig, tmp_path: Path, clock: list[datetime], **providers) -> DataService:
    return DataService(
        cfg,
        secrets=Secrets(_env_file=None),
        cache=DiskCache(tmp_path),
        providers=providers,
        news=providers.get("_news"),  # type: ignore[arg-type]
        now=lambda: clock[0],
    )


def test_second_fetch_is_served_from_cache(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices())
    clock = [NOW]
    first = _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    assert first.info.source == "network" and first.info.requests == 1

    second = _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    assert second.info.source == "cache" and second.info.requests == 0
    assert len(yahoo.calls) == 1
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_stale_cache_fetches_only_the_tail(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices())
    clock = [NOW]
    _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    clock[0] = NOW + timedelta(hours=cfg.data.cache_ttl.prices_hours + 1)
    data = _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    assert data.info.source == "cache+network"
    _, tail_start, _ = yahoo.calls[-1]
    assert tail_start == date(2026, 9, 25) - timedelta(days=7)


def test_earlier_start_fetches_only_the_head(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices())
    clock = [NOW]
    svc = _svc(cfg, tmp_path, clock, yahoo=yahoo)
    svc.ranged("yahoo", "CL=F", date(2026, 8, 1), date(2026, 9, 25), ttl_hours=12)
    frame, info = svc.ranged("yahoo", "CL=F", date(2026, 7, 1), date(2026, 9, 25), ttl_hours=12)
    assert info.source == "cache+network"
    assert yahoo.calls[-1][1:] == (date(2026, 7, 1), date(2026, 7, 31))
    assert frame.index[0] == pd.Timestamp("2026-07-01")


def test_offline_mode(cfg: AppConfig, tmp_path: Path) -> None:
    clock = [NOW]
    offline = DataService(
        cfg, Secrets(_env_file=None), DiskCache(tmp_path), offline=True, now=lambda: clock[0]
    )
    with pytest.raises(DataUnavailable, match="offline"):
        offline.prices("CL=F")
    yahoo = FakeProvider("yahoo", _prices())
    _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    clock[0] = NOW + timedelta(days=3)
    data = offline.prices("CL=F", years=0.25)
    assert data.info.source == "cache" and data.info.stale


def test_refresh_bypasses_cache(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices())
    clock = [NOW]
    _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25)
    data = _svc(cfg, tmp_path, clock, yahoo=yahoo).prices("CL=F", years=0.25, refresh=True)
    assert data.info.source == "network" and len(yahoo.calls) == 2


def test_empty_upstream_is_an_error(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices().iloc[:0])
    with pytest.raises(DataUnavailable, match="no data"):
        _svc(cfg, tmp_path, [NOW], yahoo=yahoo).prices("XYZ", years=1)


def test_unknown_ticker_gets_adhoc_instrument(cfg: AppConfig, tmp_path: Path) -> None:
    yahoo = FakeProvider("yahoo", _prices())
    data = _svc(cfg, tmp_path, [NOW], yahoo=yahoo).prices("SPY", years=0.25)
    assert data.instrument.symbol == "SPY" and data.rolls == []


def test_eia_and_cot_get_release_timestamps(cfg: AppConfig, tmp_path: Path) -> None:
    periods = pd.DatetimeIndex(pd.to_datetime(["2026-09-11", "2026-09-18"]), name="date")
    eia = FakeProvider("eia", pd.DataFrame({"value": [1.0, 2.0], "units": "MBBL"}, index=periods))
    tuesdays = pd.DatetimeIndex(pd.to_datetime(["2026-09-15", "2026-09-22"]), name="date")
    cftc = FakeProvider("cftc", pd.DataFrame({"mm_long": [1.0, 2.0]}, index=tuesdays))
    svc = _svc(cfg, tmp_path, [NOW], eia=eia, cftc=cftc)

    frame, _ = svc.eia("PET.WCESTUS1.W")
    assert frame.loc["2026-09-18", "available_at"] == pd.Timestamp("2026-09-23 14:30", tz="UTC")
    cot, _ = svc.cot("CL=F")
    assert cftc.calls[0][0] == "067651"
    assert cot.loc["2026-09-22", "available_at"] == pd.Timestamp("2026-09-25 19:30", tz="UTC")
    with pytest.raises(DataUnavailable, match="CFTC"):
        svc.cot("TTF=F")


def test_fred_availability_from_first_publication(cfg: AppConfig, tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-09-17", "2026-09-18"]), name="date")
    fred = FakeProvider(
        "fred",
        pd.DataFrame(
            {
                "value": [4.25, 4.27],
                "first_published": pd.to_datetime(["2026-09-18", "2026-09-21"]),
            },
            index=idx,
        ),
    )
    frame, _ = _svc(cfg, tmp_path, [NOW], fred=fred).fred("DGS10")
    assert frame.loc["2026-09-18", "available_at"] == pd.Timestamp("2026-09-22 04:00", tz="UTC")


def test_news_snapshot_respects_ttl(cfg: AppConfig, tmp_path: Path) -> None:
    fake = FakeNews()
    clock = [NOW]
    svc = _svc(cfg, tmp_path, clock, _news=fake)
    frame, infos = svc.news("CL=F")
    n_feeds = len(infos)
    assert fake.requests_made == n_feeds and not frame.empty
    svc.news("CL=F")
    assert fake.requests_made == n_feeds  # cached within TTL
    clock[0] = NOW + timedelta(hours=cfg.data.cache_ttl.news_hours + 1)
    svc.news("CL=F")
    assert fake.requests_made == 2 * n_feeds


def test_cli_data_commands_offline(config_copy: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    data = yaml.safe_load((config_copy / "data.yaml").read_text())
    data["cache_dir"] = str(cache_dir)
    (config_copy / "data.yaml").write_text(yaml.safe_dump(data))
    cfg = load_config(config_copy)
    yahoo = FakeProvider("yahoo", _prices())
    DataService(cfg, Secrets(_env_file=None), providers={"yahoo": yahoo}).prices("CL=F", years=0.25)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "data",
            "prices",
            "CL=F",
            "--years",
            "0.25",
            "--offline",
            "--config-dir",
            str(config_copy),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "source: cache" in res.output and "network requests: 0" in res.output
    assert "roll=" in res.output

    res = runner.invoke(app, ["data", "cache", "--config-dir", str(config_copy)])
    assert "CL=F" in res.output
    res = runner.invoke(
        app, ["data", "sql", "SELECT count(*) AS n FROM yahoo", "--config-dir", str(config_copy)]
    )
    assert res.exit_code == 0 and "n" in res.output

    res = runner.invoke(
        app, ["data", "prices", "NG=F", "--offline", "--config-dir", str(config_copy)]
    )
    assert res.exit_code == 1 and "offline" in res.output


@pytest.mark.network
def test_live_cl_ten_years_twice_hits_cache(tmp_path: Path) -> None:
    """Phase 1 acceptance, live: run with `uv run pytest -m network`."""
    cfg = load_config()
    first = DataService(cfg, cache=DiskCache(tmp_path)).prices("CL=F", years=10)
    second = DataService(cfg, cache=DiskCache(tmp_path)).prices("CL=F", years=10)
    assert first.info.source == "network" and len(first.frame) > 2400
    assert second.info.source == "cache" and second.info.requests == 0
    shutil.rmtree(tmp_path, ignore_errors=True)
