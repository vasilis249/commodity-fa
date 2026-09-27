"""Risk and performance metrics, plus position sizing with hard risk gates.

Return inputs are simple (arithmetic) daily returns unless a name says log. NaNs
(masked days) are dropped, not treated as zero. Futures returns are already excess
returns (no cash is tied up), so Sharpe/Sortino use a zero risk-free rate by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fa.analytics.models import FiniteModel
from fa.config import RiskConfig

TRADING_DAYS = 252
_EPS = 1e-12  # below this a standard deviation is floating-point noise, i.e. zero


def to_simple(log_returns: pd.Series) -> pd.Series:
    return np.expm1(log_returns)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of 1 from simple returns (NaN days leave equity unchanged)."""
    return (1 + returns.fillna(0)).cumprod()


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = float((1 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(r)) - 1


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf_annual / periods
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if len(r) > 1 and sd > _EPS else float("nan")


def downside_deviation(
    returns: pd.Series, target: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.minimum(r - target, 0.0) ** 2)) * np.sqrt(periods))


def sortino(returns: pd.Series, target: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    dd = downside_deviation(r, target, periods)
    if r.empty or not dd > _EPS:
        return float("nan")
    return float((r.mean() - target) * periods / dd)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss of the equity curve (negative number, 0 if none)."""
    eq = equity_curve(returns)
    peak = eq.cummax().clip(lower=1.0)  # the starting capital counts as a peak
    return float((eq / peak - 1).min()) if len(eq) else float("nan")


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    return annualized_return(returns, periods) / abs(mdd) if mdd < 0 else float("nan")


def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """One-period historical VaR as a positive loss fraction."""
    r = returns.dropna()
    return float(-np.quantile(r, 1 - level)) if len(r) else float("nan")


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss beyond the VaR threshold (positive fraction)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    both = pd.concat([returns, benchmark], axis=1).dropna()
    if len(both) < 3:
        return float("nan")
    var = both.iloc[:, 1].var(ddof=1)
    return float(both.cov(ddof=1).iloc[0, 1] / var) if var > 0 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else float("nan")


class PositionSize(FiniteModel):
    """Suggested max exposure (fraction of equity, long or short) and why."""

    fraction: float
    vol_target_fraction: float | None
    binding: str  # which rule set the final number
    reasons: list[str]


def position_size(
    ann_vol: float,
    history_days: int,
    limits: RiskConfig,
    strategy_drawdown: float | None = None,
) -> PositionSize:
    """Volatility-targeted size, capped by hard limits from config/risk.yaml.

    Gates, in order: enough history, drawdown stop (only when a strategy equity
    drawdown is given, e.g. in backtests; an asset's own drawdown is not a stop),
    valid vol, then vol target capped by the per-position and gross limits.
    """
    if history_days < limits.min_history_days:
        return PositionSize(
            fraction=0.0,
            vol_target_fraction=None,
            binding="min_history_days",
            reasons=[f"only {history_days} days of history (< {limits.min_history_days})"],
        )
    if strategy_drawdown is not None and strategy_drawdown <= -limits.max_drawdown_stop:
        return PositionSize(
            fraction=0.0,
            vol_target_fraction=None,
            binding="max_drawdown_stop",
            reasons=[
                f"strategy drawdown {strategy_drawdown:.1%} "
                f"beyond stop {-limits.max_drawdown_stop:.0%}"
            ],
        )
    if not ann_vol > 0 or not np.isfinite(ann_vol):
        return PositionSize(
            fraction=0.0,
            vol_target_fraction=None,
            binding="invalid_vol",
            reasons=["volatility unavailable"],
        )
    target = limits.vol_target_annual / ann_vol
    caps = {
        "vol_target": target,
        "max_position_fraction": limits.max_position_fraction,
        "max_gross_exposure": limits.max_gross_exposure,
    }
    binding = min(caps, key=lambda k: caps[k])
    return PositionSize(
        fraction=float(caps[binding]),
        vol_target_fraction=float(target),
        binding=binding,
        reasons=[
            f"vol target {limits.vol_target_annual:.0%} / realized {ann_vol:.1%} = {target:.2f}",
            f"capped by {binding}" if binding != "vol_target" else "vol target within limits",
        ],
    )
