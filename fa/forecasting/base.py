"""Model interface for the forecasting ladder.

A model returns, for each horizon, a frame indexed by origin date with one column per
quantile (`q0.1`, `q0.5`, ...) of the h-day log return, plus `p_up` = P(return > 0).

Causality contract: `fit` receives `Dataset.until(train_end)` (labels purged by
construction); `predict` receives a view without labels, and the prediction for
origin row p may only use rows <= p. `tests/unit/test_forecasting.py` checks this for
every model by comparing batch predictions with predictions from a view cut at p.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from fa.forecasting.dataset import Dataset


def qcol(q: float) -> str:
    return f"q{q:g}"


def empty_forecast(index: pd.Index, quantiles: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=[*map(qcol, quantiles), "p_up"])


def gaussian_quantiles(
    mean: np.ndarray, sigma: np.ndarray, quantiles: Sequence[float]
) -> dict[str, np.ndarray]:
    out = {qcol(q): mean + stats.norm.ppf(q) * sigma for q in quantiles}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["p_up"] = stats.norm.cdf(mean / sigma)
    return out


def sort_quantiles(frame: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    """Fix quantile crossing (independently fitted quantile models can cross)."""
    cols = [qcol(q) for q in quantiles]
    frame[cols] = np.sort(frame[cols].to_numpy(), axis=1)
    return frame


class ForecastModel(ABC):
    name: str

    def __init__(self, horizons: Sequence[int], quantiles: Sequence[float]):
        self.horizons = tuple(horizons)
        self.quantiles = tuple(quantiles)

    def fit(self, train: Dataset) -> None:  # noqa: B027 - optional for stateless models
        """Learn from `train` (labels available only where fully observed)."""

    @abstractmethod
    def predict(self, view: Dataset, positions: Sequence[int]) -> dict[int, pd.DataFrame]:
        """Forecasts made at rows `positions` of `view` (a view without labels)."""
