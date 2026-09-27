"""End-to-end forecasting on synthetic data: honesty about (no) edge, cache, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fa.config import AppConfig
from fa.forecasting import service as fsvc
from fa.forecasting.service import forecast_dataset
from tests.forecast_helpers import small_config, synthetic_dataset

FAST = ["naive", "drift", "auto_ets", "auto_arima"]


def _cfg(cfg: AppConfig, tmp: Path, **wf) -> AppConfig:
    small = small_config(cfg, **wf)
    fc = small.forecasting.model_copy(update={"leaderboard_dir": tmp / "lb"})
    return small.model_copy(update={"forecasting": fc})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_walk_gets_no_edge_claim(cfg: AppConfig, tmp_path: Path, seed: int) -> None:
    c = _cfg(cfg, tmp_path)
    ds = synthetic_dataset(c, n=800, phi=0.0, seed=seed)
    rep = forecast_dataset(ds, c, models=[*FAST, "lightgbm"])
    assert not any(r.edge for r in rep.leaderboard)
    for hf in rep.horizons:
        assert hf.model == "naive"
        assert hf.skill.verdict.startswith(f"No measurable edge at {hf.horizon}d")


def test_planted_signal_is_detected(cfg: AppConfig, tmp_path: Path) -> None:
    """AR(1) daily returns (phi=0.5): an ARIMA should beat the random walk at 1 day."""
    c = _cfg(cfg, tmp_path, eval_stride=1, max_folds=15)
    ds = synthetic_dataset(c, n=900, phi=0.5, seed=4)
    rep = forecast_dataset(ds, c, models=["naive", "auto_arima"])
    one = next(hf for hf in rep.horizons if hf.horizon == 1)
    assert one.skill.edge and one.model == "auto_arima"
    assert "beats the naive random walk" in one.skill.verdict


def test_report_shape_and_bands(cfg: AppConfig, tmp_path: Path) -> None:
    c = _cfg(cfg, tmp_path)
    ds = synthetic_dataset(c, n=700)
    rep = forecast_dataset(ds, c, models=FAST)
    assert [hf.horizon for hf in rep.horizons] == [1, 5, 20]
    for hf in rep.horizons:
        prices = [b.price for b in hf.bands]
        assert prices == sorted(prices)  # P10 <= P50 <= P90
        assert hf.p_up is not None and 0 <= hf.p_up <= 1
        assert set(hf.all_models) >= set(FAST)
    assert "ensemble" in rep.models
    assert "Not investment advice" in rep.disclaimer
    assert json.loads(rep.model_dump_json())["symbol"] == "SYN"


def test_leaderboard_is_cached(cfg: AppConfig, tmp_path: Path, monkeypatch) -> None:
    c = _cfg(cfg, tmp_path)
    ds = synthetic_dataset(c, n=700)
    first = forecast_dataset(ds, c, models=["naive", "drift"])
    assert list((tmp_path / "lb").glob("*.json"))

    def boom(*a, **k):
        raise AssertionError("walk-forward re-ran despite a cached leaderboard")

    monkeypatch.setattr(fsvc, "walk_forward", boom)
    second = forecast_dataset(ds, c, models=["naive", "drift"])
    assert second.leaderboard == first.leaderboard
    with pytest.raises(AssertionError):
        forecast_dataset(ds, c, models=["naive", "drift"], refresh=True)


def test_cli_forecast(cfg: AppConfig, tmp_path: Path, monkeypatch) -> None:
    from fa.cli import app

    c = _cfg(cfg, tmp_path)
    ds = synthetic_dataset(c, n=700)
    monkeypatch.setattr(fsvc, "build_dataset", lambda svc, symbol, horizons, as_of=None: ds)
    monkeypatch.setattr("fa.cli._load_or_exit", lambda _: c)
    res = CliRunner().invoke(app, ["forecast", "SYN", "--models", "naive,drift", "--offline"])
    assert res.exit_code == 0, res.output
    assert "LEADERBOARD" in res.output and "No measurable edge" in res.output
    res = CliRunner().invoke(app, ["forecast", "SYN", "--models", "prophet", "--offline"])
    assert res.exit_code == 1 and "unknown models" in res.output
