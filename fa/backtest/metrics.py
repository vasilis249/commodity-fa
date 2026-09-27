"""Backtest metrics, always reported next to the buy-and-hold benchmark."""

from __future__ import annotations

import math

import numpy as np

from fa.analytics import risk
from fa.analytics.models import FiniteModel
from fa.backtest.engine import BacktestResult

TRADING_DAYS = 252


class Metrics(FiniteModel):
    name: str
    start: str
    end: str
    days: int
    total_return: float | None
    cagr: float | None
    vol: float | None
    sharpe: float | None
    sharpe_ci95: tuple[float, float] | None  # Lo (2002) iid approximation
    sortino: float | None
    max_drawdown: float | None
    calmar: float | None
    hit_rate: float | None  # share of invested days with a positive return
    turnover: float | None  # traded notional per year (1.0 = one full round of equity)
    exposure: float | None  # average |position|
    trades: int
    costs_paid: float | None  # sum of cost fractions (approx. share of equity lost to costs)


def summarize(result: BacktestResult) -> Metrics:
    f = result.frame
    r = f["ret"].iloc[1:]  # day 0 has no position by construction
    n = len(r)
    years = n / TRADING_DAYS if n else float("nan")
    sharpe = risk.sharpe(r)
    se = (
        math.sqrt((1 + 0.5 * sharpe**2) / n) * math.sqrt(TRADING_DAYS)
        if n > 1 and math.isfinite(sharpe)
        else float("nan")
    )
    invested = f["position"].iloc[1:] != 0
    return Metrics(
        name=result.name,
        start=str(f.index[0].date()),
        end=str(f.index[-1].date()),
        days=n,
        total_return=float(f["equity"].iloc[-1] - 1),
        cagr=risk.annualized_return(r),
        vol=risk.annualized_vol(r),
        sharpe=sharpe,
        sharpe_ci95=(sharpe - 1.96 * se, sharpe + 1.96 * se) if math.isfinite(se) else None,
        sortino=risk.sortino(r),
        max_drawdown=risk.max_drawdown(r),
        calmar=risk.calmar(r),
        hit_rate=float((r[invested] > 0).mean()) if invested.any() else None,
        turnover=float(f["trade"].abs().sum() / years) if years and years > 0 else None,
        exposure=float(f["position"].iloc[1:].abs().mean()) if n else None,
        trades=int((f["trade"].abs() > 1e-12).sum()),
        costs_paid=float(f["cost"].sum()),
    )


def excess(strategy: Metrics, benchmark: Metrics) -> dict[str, float | None]:
    def diff(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None or not np.isfinite(a - b) else a - b

    return {
        "cagr": diff(strategy.cagr, benchmark.cagr),
        "sharpe": diff(strategy.sharpe, benchmark.sharpe),
        "max_drawdown": diff(strategy.max_drawdown, benchmark.max_drawdown),
    }
