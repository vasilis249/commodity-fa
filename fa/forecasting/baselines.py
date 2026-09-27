"""Random-walk baselines: the bar every other model must clear.

Both use the empirical distribution of past h-day log returns in a trailing window
(only returns fully observed by the origin). `naive` is the classic no-change forecast:
the distribution is re-centred on a zero median (P50 = 0, P(up) = 50%). `drift` keeps
the trailing distribution as it is, trend included.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fa.forecasting.base import ForecastModel, empty_forecast
from fa.forecasting.dataset import MIN_FRACTION, Dataset


class EmpiricalRandomWalk(ForecastModel):
    def __init__(
        self,
        horizons: Sequence[int],
        quantiles: Sequence[float],
        lookback: int,
        min_days: int,
        demean: bool,
    ):
        super().__init__(horizons, quantiles)
        self.lookback, self.min_days, self.demean = lookback, min_days, demean
        self.name = "naive" if demean else "drift"

    def predict(self, view: Dataset, positions: Sequence[int]) -> dict[int, pd.DataFrame]:
        out = {}
        for h in self.horizons:
            # sum of the h returns ending at s (known at s)
            past = view.r.rolling(h, min_periods=int(np.ceil(h * MIN_FRACTION))).sum().to_numpy()
            frame = empty_forecast(view.dates[list(positions)], self.quantiles)
            for i, p in enumerate(positions):
                window = past[max(0, p - self.lookback + 1) : p + 1]
                window = window[~np.isnan(window)]
                if len(window) < self.min_days:
                    continue
                x = window - np.median(window) if self.demean else window
                frame.iloc[i, : len(self.quantiles)] = np.quantile(x, self.quantiles)
                frame.iloc[i, -1] = 0.5 if self.demean else float((x > 0).mean())
            out[h] = frame
        return out


def naive(
    horizons: Sequence[int], quantiles: Sequence[float], lookback: int, min_days: int
) -> EmpiricalRandomWalk:
    return EmpiricalRandomWalk(horizons, quantiles, lookback, min_days, demean=True)


def drift(
    horizons: Sequence[int], quantiles: Sequence[float], lookback: int, min_days: int
) -> EmpiricalRandomWalk:
    return EmpiricalRandomWalk(horizons, quantiles, lookback, min_days, demean=False)
