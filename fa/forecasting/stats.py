"""AutoARIMA / AutoETS (statsforecast) on the log of the causal level index.

Fitted once per fold; at each origin the fitted model is re-applied (`forward`) to the
data up to that origin, so its state is the one known then. Intervals are Gaussian, so
quantiles come from the 80% interval and P(up) from the normal CDF.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from fa.forecasting.base import ForecastModel, empty_forecast, gaussian_quantiles
from fa.forecasting.dataset import Dataset

log = logging.getLogger(__name__)
Z90 = stats.norm.ppf(0.9)


class StatsModel(ForecastModel):
    def __init__(self, horizons: Sequence[int], quantiles: Sequence[float], kind: str):
        super().__init__(horizons, quantiles)
        self.kind = kind
        self.name = kind
        self._model: object | None = None

    def _new(self) -> object:
        from statsforecast.models import AutoARIMA, AutoETS

        if self.kind == "auto_arima":
            # small order search: same orders as the full search on our data, ~15x faster
            return AutoARIMA(season_length=1, max_p=2, max_q=2, approximation=True)
        return AutoETS(season_length=1)

    def fit(self, train: Dataset) -> None:
        y = np.log(train.level.to_numpy())
        self._model = self._new()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(y)  # type: ignore[attr-defined]

    def predict(self, view: Dataset, positions: Sequence[int]) -> dict[int, pd.DataFrame]:
        if self._model is None:
            raise RuntimeError("fit() first")
        y = np.log(view.level.to_numpy())
        hmax = max(self.horizons)
        rows: dict[int, list[dict[str, float]]] = {h: [] for h in self.horizons}
        for p in positions:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = self._model.forward(y=y[: p + 1], h=hmax, level=[80])  # type: ignore[attr-defined]
                mean = np.asarray(fc["mean"], dtype=float) - y[p]
                sigma = (np.asarray(fc["hi-80"], float) - np.asarray(fc["lo-80"], float)) / (
                    2 * Z90
                )
            except Exception as exc:  # a failed origin is reported as missing, not fatal
                log.warning("%s forward failed at %s: %s", self.name, view.dates[p].date(), exc)
                mean = sigma = np.full(hmax, np.nan)
            for h in self.horizons:
                q = gaussian_quantiles(
                    np.array([mean[h - 1]]), np.array([sigma[h - 1]]), self.quantiles
                )
                rows[h].append({k: float(v[0]) for k, v in q.items()})
        idx = view.dates[list(positions)]
        out = {}
        for h in self.horizons:
            frame = empty_forecast(idx, self.quantiles)
            if rows[h]:
                frame.loc[:, :] = pd.DataFrame(rows[h], index=idx)[frame.columns].to_numpy()
            out[h] = frame
        return out
