"""Synthetic series and small configs for forecasting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fa.config import AppConfig, RollRule
from fa.data.rolls import roll_adjust
from fa.data.service import adhoc_instrument
from fa.forecasting.dataset import Dataset, make_dataset


def synthetic_prices(
    n: int = 900, phi: float = 0.0, seed: int = 0, vol: float = 0.015
) -> pd.DataFrame:
    """OHLCV with AR(1) daily log returns (phi=0: a pure random walk)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, vol, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = 50 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2019-01-02", periods=n, name="date")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 2_000, n).astype(float),
        },
        index=idx,
    )


def synthetic_dataset(cfg: AppConfig, **kw) -> Dataset:
    prices = synthetic_prices(**kw)
    adjusted, _ = roll_adjust(prices, RollRule.NONE, "cme", cfg.data.rolls)
    inst = adhoc_instrument("SYN")
    return make_dataset("SYN", adjusted, inst, cfg.data.rolls, tuple(cfg.forecasting.horizons))


def small_config(cfg: AppConfig, **wf) -> AppConfig:
    """Fast settings: few folds, tiny LightGBM."""
    fc = cfg.forecasting
    walk = fc.walk_forward.model_copy(
        update={"min_train_days": 400, "max_folds": 12, "refit_every": 2, **wf}
    )
    lgbm = fc.lightgbm.model_copy(update={"n_estimators": 40, "num_threads": 1})
    base = fc.baselines.model_copy(update={"lookback_days": 300, "min_days": 100})
    fc2 = fc.model_copy(update={"walk_forward": walk, "lightgbm": lgbm, "baselines": base})
    return cfg.model_copy(update={"forecasting": fc2})
