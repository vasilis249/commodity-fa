from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fa.config import AppConfig
from fa.forecasting import evaluate as ev
from fa.forecasting.base import qcol
from fa.forecasting.service import model_factories
from tests.forecast_helpers import small_config, synthetic_dataset


def test_folds_respect_embargo_and_stride(cfg: AppConfig) -> None:
    small = small_config(cfg).forecasting
    folds = ev.plan_folds(1000, small)
    wf = small.walk_forward
    assert len(folds) == wf.max_folds
    for f in folds:
        assert f.train_end == f.test_start - 1 - wf.embargo_days
        assert all(o % wf.eval_stride == 0 and f.test_start <= o <= f.test_end for o in f.origins)
    assert folds[-1].test_end == 999


@pytest.mark.parametrize("name", ["naive", "drift", "auto_arima", "auto_ets", "lightgbm"])
def test_models_only_use_rows_up_to_each_origin(cfg: AppConfig, name: str) -> None:
    """Batch prediction from a long view == prediction from a view cut at each origin."""
    small = small_config(cfg)
    ds = synthetic_dataset(small, n=700)
    model = model_factories(small.forecasting, [name])[name]()
    model.fit(ds.until(450))
    positions = [500, 560, 620]
    batch = model.predict(ds.until(699, with_labels=False), positions)
    for p in positions:
        single = model.predict(ds.until(p, with_labels=False), [p])
        for h in ds.horizons:
            pd.testing.assert_frame_equal(
                batch[h].loc[[ds.dates[p]]], single[h], check_exact=False, check_freq=False
            )


def test_naive_is_a_no_change_forecast(cfg: AppConfig) -> None:
    small = small_config(cfg)
    ds = synthetic_dataset(small, n=600)
    out = model_factories(small.forecasting, ["naive"])["naive"]().predict(
        ds.until(599, False), [599]
    )
    for frame in out.values():
        row = frame.iloc[0]
        assert row[qcol(0.5)] == pytest.approx(0.0, abs=1e-12)
        assert row["p_up"] == 0.5
        assert row[qcol(0.1)] < 0 < row[qcol(0.9)]


def test_pinball_and_dm_test() -> None:
    y = np.array([0.0, 1.0])
    assert ev.pinball(y, np.array([0.0, 0.0]), 0.9).tolist() == pytest.approx([0.0, 0.9])
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 400)
    assert ev.diebold_mariano(noise - 0.5, lag=0) < 0.001  # clearly lower loss
    assert ev.diebold_mariano(noise + 0.5, lag=0) > 0.999
    assert np.isnan(ev.diebold_mariano(np.zeros(5), lag=0))


def test_non_overlapping_origins() -> None:
    pos = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    assert ev.non_overlapping(pd.DatetimeIndex([]), pos, 20).tolist() == [0, 4, 8]
    assert ev.non_overlapping(pd.DatetimeIndex([]), pos, 1).tolist() == list(range(9))


def test_ensemble_combine_ignores_missing_models() -> None:
    arr = np.array([[[1.0, 2.0]], [[np.nan, 4.0]]])  # (models, rows, cols)
    out = ev.combine(arr, np.array([0.25, 0.75]))
    assert out.tolist() == [[1.0, 0.25 * 2 + 0.75 * 4]]


def test_ensemble_weights_use_only_earlier_folds(cfg: AppConfig) -> None:
    fold_ids = np.repeat(np.arange(6), 4)
    losses = {"a": np.ones(24), "b": np.full(24, 2.0)}
    w = ev._weights(losses, fold_ids, 5, cfg.forecasting)
    losses["b"][fold_ids >= 5] = 1000.0  # the current/future folds must not matter
    assert ev._weights(losses, fold_ids, 5, cfg.forecasting) == w
    assert w["a"] == pytest.approx(2 / 3)


def test_unknown_model_and_missing_chronos(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="unknown models"):
        model_factories(cfg.forecasting, ["prophet"])
    notes: list[str] = []
    f = model_factories(cfg.forecasting, ["chronos2", "lightgbm"], notes)
    assert "naive" in f  # the baseline is always evaluated
    from fa.forecasting.foundation import chronos_available

    if not chronos_available():
        assert "chronos2" not in f and any("chronos2 skipped" in n for n in notes)


def test_chronos_output_shapes() -> None:
    from fa.forecasting.foundation import _to_numpy

    class FakeTensor:
        def __init__(self, a: np.ndarray):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self.a

    per_series = [FakeTensor(np.ones((1, 20, 3))) for _ in range(4)]  # Chronos-2 style
    assert _to_numpy(per_series).shape == (4, 20, 3)
    assert _to_numpy(FakeTensor(np.ones((4, 20, 3)))).shape == (4, 20, 3)  # older style


def test_holm_adjustment() -> None:
    assert ev.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ev.holm([0.5]) == [0.5]
