"""End-to-end numeric snapshot on fixtures: schema, point-in-time, no look-ahead."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from fa.analytics.snapshot import Snapshot, build_snapshot
from fa.cli import app
from fa.config import AppConfig, Secrets, load_config
from fa.data.cache import DiskCache
from fa.data.service import DataService
from tests.conftest import FIXTURES, load_yahoo_fixture

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


class KeyedProvider:
    """Serves a different frame per key; records calls."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: list[str] = []

    @property
    def requests_made(self) -> int:
        return len(self.calls)

    def fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(key)
        return self.frames[key].loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _weekly(freq: str, base: float, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2017-01-01", "2026-09-25", freq=freq, name="date")
    rng = np.random.default_rng(seed)
    season = 10 * np.sin(2 * np.pi * idx.dayofyear / 365)
    return pd.DataFrame(
        {"value": base + season + rng.normal(0, 2, len(idx)), "units": "MBBL"}, index=idx
    )


def _cot() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2026-09-25", freq="W-TUE", name="date")
    n = len(idx)
    rng = np.random.default_rng(7)
    cols = [
        "open_interest",
        "prod_long",
        "prod_short",
        "swap_long",
        "swap_short",
        "mm_long",
        "mm_short",
    ]
    frame = pd.DataFrame(
        {c: rng.integers(100_000, 900_000, n).astype(float) for c in cols}, index=idx
    )
    frame["open_interest"] = 2_000_000.0
    return frame


def _curve_frames() -> dict[str, pd.DataFrame]:
    settles = {
        "CLX26": 92.41,
        "CLZ26": 88.71,
        "CLF27": 86.01,
        "CLG27": 83.97,
        "CLH27": 82.46,
        "CLJ27": 81.12,
    }
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-25")], name="date")
    return {f"{k}.NYM": pd.DataFrame({"close": [v]}, index=idx) for k, v in settles.items()}


def _prices(cut: str | None = None) -> dict[str, pd.DataFrame]:
    cl = pd.read_csv(FIXTURES / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
    cl.index.name = "date"
    frames = {
        "CL=F": cl,
        "RB=F": load_yahoo_fixture("RB=F", "2026"),
        "HO=F": load_yahoo_fixture("HO=F", "2026"),
    }
    frames |= _curve_frames()
    if cut:
        frames = {k: v.loc[:cut] for k, v in frames.items()}
    return frames


def _service(cfg: AppConfig, tmp: Path, cut: str | None = None) -> DataService:
    eia = {
        sid: _weekly("W-FRI", 400 + i * 50, i)
        for i, sid in enumerate(cfg.data.instrument("CL=F").eia_series.values())
    }  # type: ignore[union-attr]
    cftc = {"067651": _cot()}
    if cut:
        eia = {k: v.loc[:cut] for k, v in eia.items()}
        cftc = {k: v.loc[:cut] for k, v in cftc.items()}
    return DataService(
        cfg,
        secrets=Secrets(_env_file=None),
        cache=DiskCache(tmp),
        providers={
            "yahoo": KeyedProvider(_prices(cut)),
            "eia": KeyedProvider(eia),
            "cftc": KeyedProvider(cftc),
        },
        now=lambda: NOW,
    )


def test_snapshot_is_complete_and_schema_valid(cfg: AppConfig, tmp_path: Path) -> None:
    snap = build_snapshot(_service(cfg, tmp_path), "CL=F")
    assert snap.as_of == date(2026, 9, 25)
    assert snap.price.last_close == pytest.approx(92.41, abs=0.01)
    assert snap.technicals.rsi14 is not None and 0 <= snap.technicals.rsi14 <= 100
    assert len(snap.inventories) == 4
    assert snap.cot is not None and snap.crack is not None
    assert snap.curve is not None and snap.curve.metrics.structure == "backwardation"
    assert 0 <= snap.risk.position.fraction <= cfg.risk.max_position_fraction
    assert "Not investment advice" in snap.disclaimer
    again = Snapshot.model_validate_json(snap.model_dump_json())
    assert again == snap


def test_snapshot_respects_release_times(cfg: AppConfig, tmp_path: Path) -> None:
    # Tue 22 Sep 2026: the week to Fri 18 Sep is released Wed 23 Sep -> not yet public
    snap = build_snapshot(
        _service(cfg, tmp_path), "CL=F", as_of=date(2026, 9, 22), include_curve=False
    )
    assert {inv.period for inv in snap.inventories} == {date(2026, 9, 11)}
    assert all(inv.released_at <= snap.decision_time for inv in snap.inventories)
    # COT for Tue 15 Sep was released Fri 18 Sep; Tue 22 Sep's report is not out yet
    assert snap.cot is not None and snap.cot.report_date == date(2026, 9, 15)


def test_snapshot_has_no_look_ahead(cfg: AppConfig, tmp_path: Path) -> None:
    """A past snapshot is identical whether or not the providers hold later data."""
    as_of = date(2026, 6, 30)
    full = build_snapshot(_service(cfg, tmp_path / "a"), "CL=F", as_of=as_of, include_curve=False)
    cut = build_snapshot(
        _service(cfg, tmp_path / "b", cut="2026-06-30"), "CL=F", as_of=as_of, include_curve=False
    )
    ignore = {"generated_at", "sources"}
    assert full.model_dump(exclude=ignore) == cut.model_dump(exclude=ignore)


def test_past_snapshot_omits_live_curve(cfg: AppConfig, tmp_path: Path) -> None:
    snap = build_snapshot(_service(cfg, tmp_path), "CL=F", as_of=date(2026, 3, 2))
    assert snap.curve is None
    assert any("only the live curve" in n for n in snap.notes)


def test_cli_analyze_json_offline(config_copy: Path, tmp_path: Path) -> None:
    data = yaml.safe_load((config_copy / "data.yaml").read_text())
    data["cache_dir"] = str(tmp_path / "cache")
    (config_copy / "data.yaml").write_text(yaml.safe_dump(data))
    cfg = load_config(config_copy)
    svc = _service(cfg, tmp_path / "cache")
    build_snapshot(svc, "CL=F")  # warms the cache used by the CLI

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["analyze", "CL=F", "--no-llm", "--offline", "--json", "--config-dir", str(config_copy)],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["symbol"] == "CL=F" and payload["disclaimer"]

    res = runner.invoke(
        app, ["analyze", "CL=F", "--no-llm", "--offline", "--config-dir", str(config_copy)]
    )
    assert (
        res.exit_code == 0 and "TECHNICALS" in res.output and "Not investment advice" in res.output
    )
