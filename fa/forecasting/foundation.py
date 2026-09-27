"""Chronos-2 zero-shot forecasts (optional: `uv sync --extra foundation`).

The model sees the causal level index (last `context_length` values up to the origin)
and returns quantiles of future levels; these become log-return quantiles. P(up) is
interpolated from a dense quantile grid. Nothing is trained, so `fit` only loads the
pipeline once.

TimesFM is deliberately not wired in: its latest weights are non-commercial.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fa.config import ChronosSettings
from fa.forecasting.base import ForecastModel, empty_forecast, qcol
from fa.forecasting.dataset import Dataset

GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _to_numpy(quantiles: object) -> np.ndarray:
    """(batch, h, q) array from Chronos-2 (list of (variates, h, q)) or older (tensor)."""
    if isinstance(quantiles, list):
        return np.stack([np.asarray(t.detach().cpu())[0] for t in quantiles])
    return np.asarray(quantiles.detach().cpu())  # type: ignore[attr-defined]


class ChronosModel(ForecastModel):
    name = "chronos2"

    def __init__(
        self,
        horizons: Sequence[int],
        quantiles: Sequence[float],
        settings: ChronosSettings,
        pipeline: object | None = None,
    ):
        super().__init__(horizons, quantiles)
        self.settings = settings
        self._pipeline = pipeline  # injectable for tests

    def fit(self, train: Dataset) -> None:
        if self._pipeline is None:
            from chronos import BaseChronosPipeline

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.settings.model_id, device_map=self.settings.device
            )

    def predict(self, view: Dataset, positions: Sequence[int]) -> dict[int, pd.DataFrame]:
        import torch

        if self._pipeline is None:
            self.fit(view)
        pipeline: Any = self._pipeline
        level = view.level.to_numpy()
        ctx = self.settings.context_length
        inputs = [
            torch.tensor(level[max(0, p - ctx + 1) : p + 1], dtype=torch.float32) for p in positions
        ]
        levels = sorted(set(GRID) | set(self.quantiles))
        q, _ = pipeline.predict_quantiles(
            inputs, prediction_length=max(self.horizons), quantile_levels=levels
        )
        arr = _to_numpy(q)  # (batch, h, len(levels))
        base = level[list(positions)][:, None]
        idx = view.dates[list(positions)]
        out = {}
        for h in self.horizons:
            logq = np.log(np.maximum(arr[:, h - 1, :], 1e-12) / base)  # (batch, levels)
            frame = empty_forecast(idx, self.quantiles)
            for q_level in self.quantiles:
                frame[qcol(q_level)] = logq[:, levels.index(q_level)]
            # P(up) = 1 - F(0): interpolate the level CDF at zero log return
            frame["p_up"] = [
                1 - float(np.interp(0.0, row, levels, left=0.0, right=1.0)) for row in logq
            ]
            out[h] = frame
        return out
