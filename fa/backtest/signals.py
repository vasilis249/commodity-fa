"""Signals: target weights decided at each close from data up to that close.

All signals read the causal series of `fa.forecasting.dataset.Dataset` (calendar-masked
returns and the level index anchored at the first bar), never `close_adj`, whose roll
mask and anchoring use hindsight. `tests/unit/test_backtest.py` checks causality with
`assert_causal`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fa.analytics.indicators import realized_vol, sma
from fa.config import AppConfig, ForecastSignalSettings, SMASettings
from fa.forecasting.dataset import Dataset
from fa.forecasting.evaluate import EvalResult, walk_forward
from fa.forecasting.service import model_factories


def sma_crossover(level: pd.Series, s: SMASettings, long_short: bool) -> pd.Series:
    """+1 when SMA(fast) > SMA(slow); -1 (or 0 if long-only) when below; NaN in warm-up."""
    fast, slow = sma(level, s.fast), sma(level, s.slow)
    down = -1.0 if long_short else 0.0
    return pd.Series(np.where(fast > slow, 1.0, down), index=level.index).where(slow.notna())


def forecast_targets(
    ds: Dataset, cfg: AppConfig, s: ForecastSignalSettings, long_short: bool
) -> tuple[pd.Series, EvalResult]:
    """Targets at walk-forward origins from P(up) at horizon `s.horizon`; NaN between
    origins (the engine holds the last target). Uses every fold, not only the recent ones."""
    wf = cfg.forecasting.walk_forward.model_copy(update={"max_folds": 10**6})
    fc = cfg.forecasting.model_copy(update={"walk_forward": wf})
    factories = model_factories(fc, list(s.members))
    ev = walk_forward(ds, factories, fc)
    if s.model not in ev.predictions:
        raise ValueError(f"model {s.model!r} not evaluated (members: {list(ev.predictions)})")
    p_up = ev.predictions[s.model][s.horizon]["p_up"]
    down = -1.0 if long_short else 0.0
    decided = np.where(p_up >= s.up_threshold, 1.0, np.where(p_up <= s.down_threshold, down, 0.0))
    decided = np.where(p_up.isna(), np.nan, decided)
    targets = pd.Series(np.nan, index=ds.dates)
    targets.loc[p_up.index] = decided
    return targets, ev


def vol_targeted(targets: pd.Series, r: pd.Series, vol_target: float, cap: float) -> pd.Series:
    """Scale targets to an annual vol target using trailing 20-day vol (causal)."""
    vol = realized_vol(r, 20, min_fraction=0.6)
    scale = (vol_target / vol).clip(upper=cap).ffill()
    held = targets.ffill()
    return (held * scale).where(held.notna())


def first_valid(targets: pd.Series) -> pd.Timestamp | None:
    valid = targets.dropna()
    return valid.index[0] if len(valid) else None
