"""Run a strategy backtest against buy-and-hold over the same window, and save it."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fa.backtest.engine import BacktestResult, CostModel, buy_and_hold, pnl_bars, run_backtest
from fa.backtest.metrics import Metrics, edge_pvalue, excess, summarize
from fa.backtest.signals import first_valid, forecast_targets, sma_crossover, vol_targeted
from fa.config import AppConfig
from fa.data.calendars import calendar_for_exchange
from fa.data.rolls import expiries
from fa.data.service import DataService
from fa.forecasting.dataset import Dataset, build_dataset

DISCLAIMER = "Paper backtest for research only. Past simulated results are not investment advice."
Strategy = Literal["sma", "forecast", "agent"]


class BacktestReport(BaseModel):
    symbol: str
    strategy: str
    params: dict[str, Any]
    strategy_metrics: Metrics
    benchmark_metrics: Metrics
    excess: dict[str, float | None]
    edge_p_value: float | None  # one-sided, strategy mean daily return > buy-and-hold's
    notes: list[str]
    results_file: str | None = None
    generated_at: datetime
    disclaimer: str = DISCLAIMER


def bars_and_rolls(svc: DataService, symbol: str, ds: Dataset) -> tuple[pd.DataFrame, pd.Series]:
    """P&L bars (`pnl_bars`) and roll sessions (expiry calendar, known in advance)."""
    frame = svc.prices(symbol, as_of=ds.dates[-1].date()).frame.reindex(ds.dates)
    bars = pnl_bars(frame)
    inst = svc.instrument(symbol)
    rolls = pd.Series(False, index=ds.dates)
    if inst.roll_rule.value != "none":
        cal = calendar_for_exchange(inst.exchange)
        for e in expiries(inst.roll_rule, ds.dates[0].date(), ds.dates[-1].date(), cal).index:
            pos = ds.dates.searchsorted(e, side="right")  # first session after expiry
            if pos < len(ds.dates):
                rolls.iloc[pos] = True
    return bars, rolls


def compare(
    bars: pd.DataFrame, targets: pd.Series, rolls: pd.Series, costs: CostModel, name: str
) -> tuple[BacktestResult, BacktestResult]:
    """Strategy and buy-and-hold over the same window, from the first valid target."""
    start = first_valid(targets)
    if start is None:
        raise ValueError(f"{name}: the signal never produced a target")
    window = bars.loc[start:]
    strat = run_backtest(window, targets.loc[start:], costs, name, rolls.loc[start:])
    bench = buy_and_hold(window, costs, rolls.loc[start:])
    return strat, bench


def accounting_notes(bars: pd.DataFrame, strat: BacktestResult, bench: BacktestResult) -> list[str]:
    """What the P&L could not credit inside the backtest window."""
    z = bars["zeroed"].reindex(strat.frame.index).iloc[1:]
    held = strat.frame["position"].iloc[1:] != 0
    notes = []
    n_roll = int((z == "overnight").sum())
    if n_roll:
        notes.append(
            f"{n_roll} roll sessions: the overnight gap (contract splice) earns nothing; the "
            f"intraday move is credited ({int(((z == 'overnight') & held).sum())} while invested)."
        )
    bad = z == "both"
    if bad.any():
        reasons = bars["mask_reason"].reindex(z.index)[bad]
        np_days = [str(d.date()) for d in reasons.index[reasons == "nonpositive_price"]]
        if np_days:
            notes.append(
                "WARNING: non-positive prices on "
                + ", ".join(np_days)
                + f" (strategy {'invested' if held[bad].any() else 'flat'}, buy-and-hold "
                f"{'invested' if (bench.frame['position'].iloc[1:][bad] != 0).any() else 'flat'}"
                "): no return is defined, so these sessions earn nothing; a real position "
                "could have lost far more."
            )
        n_missing = int((reasons == "missing").sum())
        if n_missing:
            notes.append(f"{n_missing} sessions with missing prices earn nothing.")
    return notes


def run_strategy(
    svc: DataService,
    symbol: str,
    strategy: Strategy,
    as_of: date | None = None,
    ds: Dataset | None = None,
    save: bool = True,
    **agent_kw: Any,
) -> BacktestReport:
    cfg: AppConfig = svc.cfg
    bt = cfg.backtest
    ds = ds or build_dataset(svc, symbol, tuple(cfg.forecasting.horizons), as_of=as_of)
    bars, rolls = bars_and_rolls(svc, symbol, ds)
    costs = CostModel.from_config(cfg.risk.backtest)
    notes: list[str] = [
        f"Costs: {cfg.risk.backtest.cost_bps_per_side:g} bps per side + "
        f"{cfg.risk.backtest.slippage_bps_per_side:g} bps slippage; "
        f"roll cost {cfg.risk.backtest.roll_cost_bps:g} bps per roll.",
        "Targets decided at the close, executed at the next open; P&L follows the held "
        "front-month contract (contract splices earn nothing).",
    ]
    params: dict[str, Any] = {"long_short": bt.long_short, "vol_target": bt.vol_target}
    if strategy == "sma":
        targets = sma_crossover(ds.level, bt.sma, bt.long_short)
        params |= bt.sma.model_dump()
    elif strategy == "forecast":
        targets, ev = forecast_targets(ds, cfg, bt.forecast, bt.long_short)
        params |= bt.forecast.model_dump()
        notes.append(
            f"Forecast signal from walk-forward predictions ({len(ev.folds)} folds); each "
            "prediction uses only data up to its origin. Thresholds are fixed in config, "
            "not tuned on this backtest."
        )
    elif strategy == "agent":
        from fa.backtest.agent_signal import agent_targets

        targets, agent_notes, agent_params = agent_targets(svc, symbol, ds, **agent_kw)
        notes += agent_notes
        params |= agent_params
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if bt.vol_target:
        targets = vol_targeted(
            targets, ds.r, cfg.risk.vol_target_annual, cfg.risk.max_gross_exposure
        )
    strat, bench = compare(bars, targets, rolls, costs, strategy)
    sm, bm = summarize(strat), summarize(bench)
    notes += accounting_notes(bars, strat, bench)
    p = edge_pvalue(strat, bench, bt.significance_block)
    alpha = cfg.forecasting.skill.significance_alpha
    if not np.isfinite(p) or p >= alpha:
        shown = f"p={p:.2f}" if np.isfinite(p) else "too little data to test"
        notes.insert(
            0,
            f"No measurable edge vs buy-and-hold ({shown}; one-sided batch-means test on "
            f"daily return differences, {bt.significance_block}-session blocks).",
        )
    else:
        notes.insert(
            0,
            f"Mean daily return above buy-and-hold (p={p:.3f}, one-sided batch-means test). "
            "One backtest; not adjusted for other strategies or settings tried.",
        )
    if sm.exposure is not None and sm.exposure < 0.05:
        notes.append(
            f"The signal was invested only {sm.exposure:.1%} of the time on average; its "
            "metrics mostly reflect holding cash (thresholds are fixed in config, not tuned)."
        )
    report = BacktestReport(
        symbol=symbol,
        strategy=strategy,
        params=params,
        strategy_metrics=sm,
        benchmark_metrics=bm,
        excess=excess(sm, bm),
        edge_p_value=float(p) if np.isfinite(p) else None,
        notes=notes,
        generated_at=datetime.now(UTC),
    )
    if save:
        report.results_file = str(_save(cfg, report, strat, bench))
    return report


def _save(
    cfg: AppConfig, report: BacktestReport, strat: BacktestResult, bench: BacktestResult
) -> Path:
    root = cfg.backtest.results_dir
    root = root if root.is_absolute() else cfg.config_dir.parent / root
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{report.symbol.replace('=', '_')}_{report.strategy}_{report.strategy_metrics.end}"
    curves = pd.DataFrame(
        {
            "strategy_equity": strat.equity,
            "benchmark_equity": bench.equity,
            "position": strat.frame["position"],
            "cost": strat.frame["cost"],
        }
    )
    curves.to_csv(root / f"{stem}.csv", float_format="%.8g")
    path = root / f"{stem}.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return path
