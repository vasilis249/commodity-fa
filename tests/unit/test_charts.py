"""Plotly figure builders: structure, dates and the forecast-skill guardrail."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fa.charts import (
    cost_by_agent_figure,
    equity_figure,
    fan_chart,
    horizon_dates,
    leaderboard_figure,
    price_figure,
    skill_caption,
)
from fa.config import AppConfig
from fa.forecasting.service import forecast_dataset
from tests.forecast_helpers import small_config, synthetic_dataset


@pytest.fixture(scope="module")
def fc_report():
    from fa.config import load_config

    small = small_config(load_config())
    ds = synthetic_dataset(small, n=600)
    return ds, forecast_dataset(ds, small, models=["naive", "drift"])


def test_fan_chart_always_carries_the_skill_line(fc_report) -> None:
    ds, rep = fc_report
    fig = fan_chart(ds.close, rep.horizons, rep.as_of, "t")
    notes = [a.text for a in fig.layout.annotations]
    assert len(notes) == len(rep.horizons)
    for h, text in zip(rep.horizons, notes, strict=True):
        assert text == skill_caption(h)
        assert "skill vs random walk" in text and "band coverage" in text and "DM p" in text
    names = [t.name for t in fig.data]
    assert "P10 to P90" in names and "median (P50)" in names


def test_fan_chart_band_is_anchored_at_the_last_close(fc_report) -> None:
    ds, rep = fc_report
    fig = fan_chart(ds.close, rep.horizons, rep.as_of, "t")
    band = next(t for t in fig.data if t.name == "P10 to P90")
    last = float(ds.close.iloc[-1])
    n = len(rep.horizons) + 1
    assert band.y[0] == pytest.approx(last) and band.y[-1] == pytest.approx(last)
    p90 = [b.price for h in rep.horizons for b in h.bands if b.quantile == 0.9]
    assert list(band.y[1:n]) == pytest.approx(p90)
    assert list(band.x[1:n]) == horizon_dates(rep.as_of, [h.horizon for h in rep.horizons])


def test_fan_chart_needs_horizons(fc_report) -> None:
    ds, rep = fc_report
    with pytest.raises(ValueError):
        fan_chart(ds.close, [], rep.as_of, "t")


def test_horizon_dates_skip_weekends() -> None:
    fri = date(2026, 9, 25)
    assert horizon_dates(fri, [1, 5]) == [pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-02")]


def test_skill_caption_marks_edges(fc_report) -> None:
    _, rep = fc_report
    h = rep.horizons[0]
    edged = h.model_copy(update={"skill": h.skill.model_copy(update={"edge": True})})
    assert "EDGE vs random walk" in skill_caption(edged)
    assert "no measurable edge" in skill_caption(
        h.model_copy(update={"skill": h.skill.model_copy(update={"edge": False})})
    )


def test_leaderboard_figure_colors_edges() -> None:
    rows = pd.DataFrame(
        {
            "model": ["naive", "a", "b"],
            "horizon": [5, 5, 5],
            "skill": [0.0, 0.02, -0.01],
            "edge": [False, True, False],
            "coverage": [0.8, 0.8, 0.8],
            "dm_p_adj": [None, 0.01, 0.9],
            "n": [100, 100, 100],
        }
    )
    fig = leaderboard_figure(rows, 5)
    bar = fig.data[0]
    colors = dict(zip(bar.y, bar.marker.color, strict=True))
    assert colors["a"] == "#2e7d32" and colors["b"] == "#c62828"


def test_price_figure(cfg: AppConfig) -> None:
    from fa.data.rolls import adjusted_returns
    from tests.conftest import FIXTURES

    prices = pd.read_csv(FIXTURES / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
    frame = adjusted_returns(prices, [])
    fig = price_figure(frame, "CL=F", rolls=None, years=1)
    names = [t.name for t in fig.data]
    assert names[:2] == ["quoted front month", "roll-adjusted close"]
    assert "SMA 50" in names and "SMA 200" in names
    sma200 = next(t for t in fig.data if t.name == "SMA 200")
    assert np.isfinite(np.asarray(sma200.y, dtype=float)).all()  # no warm-up gap in window


def test_equity_and_cost_figures() -> None:
    idx = pd.bdate_range("2024-01-01", periods=50)
    curves = pd.DataFrame(
        {
            "strategy_equity": np.linspace(1, 1.4, 50),
            "benchmark_equity": np.linspace(1, 0.7, 50),
            "position": [0.0, 1.0] * 25,
            "cost": 0.0,
        },
        index=idx,
    )
    fig = equity_figure(curves, "sma")
    assert {t.name for t in fig.data} >= {"sma", "buy & hold", "position"}
    assert 1 in fig.layout.yaxis.tickvals
    fig2 = cost_by_agent_figure({"a": 0.1, "b": 0.3})
    assert list(fig2.data[0].y) == ["a", "b"]
