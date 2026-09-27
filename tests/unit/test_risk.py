from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fa.analytics import risk
from fa.config import AppConfig


def test_annualized_return_and_vol_closed_form() -> None:
    r = pd.Series([0.001] * 252)
    assert risk.annualized_return(r) == pytest.approx(1.001**252 - 1)
    assert risk.annualized_vol(r) == pytest.approx(0.0)
    assert np.isnan(risk.sharpe(r))  # zero volatility


def test_sharpe_sortino_known_sample() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    mean, sd = r.mean(), r.std(ddof=1)
    assert risk.sharpe(r) == pytest.approx(mean / sd * np.sqrt(252))
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    assert risk.downside_deviation(r) == pytest.approx(dd)
    assert risk.sortino(r) == pytest.approx(mean * 252 / dd)


def test_max_drawdown_and_calmar() -> None:
    r = pd.Series([0.10, -0.20, 0.05])  # equity 1.1, 0.88, 0.924
    assert risk.max_drawdown(r) == pytest.approx(-0.20)
    assert risk.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)  # from the start
    assert risk.max_drawdown(pd.Series([0.01, 0.02])) == 0.0
    assert risk.calmar(r) == pytest.approx(risk.annualized_return(r) / 0.20)


def test_var_cvar() -> None:
    r = pd.Series(np.linspace(-0.10, 0.09, 20))  # -0.10, -0.09, ..., 0.09
    assert risk.var_historical(r, 0.90) == pytest.approx(-np.quantile(r, 0.10))
    assert risk.cvar_historical(r, 0.90) == pytest.approx(0.095)  # mean of -0.10, -0.09


def test_beta_and_hit_rate() -> None:
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert risk.beta(2 * b, b) == pytest.approx(2.0)
    assert risk.hit_rate(pd.Series([0.1, -0.1, 0.2, 0.0, np.nan])) == pytest.approx(2 / 3)


def test_nan_days_are_ignored_not_zeroed() -> None:
    r = pd.Series([0.01, np.nan, -0.01, 0.02])
    assert risk.annualized_vol(r) == pytest.approx(
        pd.Series([0.01, -0.01, 0.02]).std() * np.sqrt(252)
    )


def test_position_size_gates(cfg: AppConfig) -> None:
    limits = cfg.risk
    ok = risk.position_size(0.40, 1000, limits)
    assert ok.fraction == pytest.approx(
        min(limits.vol_target_annual / 0.40, limits.max_position_fraction)
    )
    low_vol = risk.position_size(0.05, 1000, limits)
    assert (
        low_vol.fraction == limits.max_position_fraction
        and low_vol.binding == "max_position_fraction"
    )
    assert risk.position_size(0.4, 10, limits).binding == "min_history_days"
    assert risk.position_size(float("nan"), 1000, limits).fraction == 0
    stopped = risk.position_size(0.4, 1000, limits, strategy_drawdown=-0.5)
    assert stopped.fraction == 0 and stopped.binding == "max_drawdown_stop"
    for size in (ok, low_vol):
        assert 0 <= size.fraction <= min(limits.max_position_fraction, limits.max_gross_exposure)
