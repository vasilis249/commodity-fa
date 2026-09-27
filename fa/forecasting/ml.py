"""LightGBM on engineered causal features: one quantile regressor per quantile and a
binary classifier for P(up), per horizon."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fa.config import LightGBMSettings
from fa.forecasting.base import ForecastModel, empty_forecast, qcol, sort_quantiles
from fa.forecasting.dataset import Dataset

MIN_TRAIN_ROWS = 300


class LightGBMModel(ForecastModel):
    name = "lightgbm"

    def __init__(
        self, horizons: Sequence[int], quantiles: Sequence[float], settings: LightGBMSettings
    ):
        super().__init__(horizons, quantiles)
        self.settings = settings
        self._models: dict[int, dict[str, object]] = {}
        self.feature_names: list[str] = []

    def _params(self) -> dict[str, object]:
        s = self.settings
        return {
            "learning_rate": s.learning_rate,
            "num_leaves": s.num_leaves,
            "min_data_in_leaf": s.min_child_samples,
            "bagging_fraction": s.subsample,
            "bagging_freq": 1,
            "feature_fraction": s.colsample_bytree,
            "lambda_l2": s.reg_lambda,
            "seed": s.seed,
            "num_threads": s.num_threads,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }

    def _train(self, x: pd.DataFrame, y: pd.Series, **objective: object) -> object:
        import lightgbm as lgb

        params = {**self._params(), **objective}
        return lgb.train(params, lgb.Dataset(x, y), num_boost_round=self.settings.n_estimators)

    def fit(self, train: Dataset) -> None:
        x = train.features
        self.feature_names = list(x.columns)
        self._models = {}
        for h in self.horizons:
            y = train.label(h)
            rows = y.notna().to_numpy()
            if rows.sum() < MIN_TRAIN_ROWS:
                continue
            xh, yh = x[rows], y[rows]
            models: dict[str, object] = {
                qcol(q): self._train(xh, yh, objective="quantile", alpha=q) for q in self.quantiles
            }
            up = (yh > 0).astype(int)
            if up.nunique() == 2:
                models["p_up"] = self._train(xh, up, objective="binary")
            self._models[h] = models

    def predict(self, view: Dataset, positions: Sequence[int]) -> dict[int, pd.DataFrame]:
        x = view.features.iloc[list(positions)][self.feature_names]
        out = {}
        for h in self.horizons:
            frame = empty_forecast(x.index, self.quantiles)
            models = self._models.get(h)
            if models:
                for key, booster in models.items():
                    frame[key] = booster.predict(x)  # type: ignore[attr-defined]
                frame = sort_quantiles(frame, self.quantiles)
            out[h] = frame
        return out

    def feature_importance(self, h: int) -> pd.Series:
        m = self._models.get(h, {}).get(qcol(0.5))
        if m is None:
            return pd.Series(dtype=float)
        gain = m.feature_importance(importance_type="gain")  # type: ignore[attr-defined]
        imp = np.asarray(gain, dtype=np.float64)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
