"""Forecast service: evaluate the model ladder, then forecast from today.

1. Build the causal dataset.
2. Walk-forward evaluation -> leaderboard (cached on disk, keyed by symbol, last data
   date, config and model list).
3. Fit each model on all data and forecast from the last session.
4. For each horizon, show the best model *with a statistically significant edge over
   the naive random walk*. If none has one, show the random walk and say plainly that
   there is no measurable edge.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fa.analytics.models import FiniteModel
from fa.config import AppConfig, ForecastingConfig, RollRule
from fa.data.service import DataService
from fa.forecasting.base import ForecastModel, qcol
from fa.forecasting.baselines import drift, naive
from fa.forecasting.dataset import Dataset, build_dataset
from fa.forecasting.evaluate import (
    AUTO_SELECT,
    ENSEMBLE,
    EvalResult,
    LeaderRow,
    ModelFactory,
    combine,
    walk_forward,
)
from fa.forecasting.foundation import ChronosModel, chronos_available
from fa.forecasting.ml import LightGBMModel
from fa.forecasting.stats import StatsModel

log = logging.getLogger(__name__)
DISCLAIMER = "Research output, paper trading only. Not investment advice."
LADDER = ("naive", "drift", "auto_arima", "auto_ets", "lightgbm", "chronos2")


def model_factories(
    cfg: ForecastingConfig, only: list[str] | None = None, notes: list[str] | None = None
) -> dict[str, ModelFactory]:
    hs, qs = tuple(cfg.horizons), tuple(cfg.quantiles)
    b = cfg.baselines
    registry: dict[str, ModelFactory] = {
        "naive": lambda: naive(hs, qs, b.lookback_days, b.min_days),
        "drift": lambda: drift(hs, qs, b.lookback_days, b.min_days),
        "auto_arima": lambda: StatsModel(hs, qs, "auto_arima"),
        "auto_ets": lambda: StatsModel(hs, qs, "auto_ets"),
        "lightgbm": lambda: LightGBMModel(hs, qs, cfg.lightgbm),
        "chronos2": lambda: ChronosModel(hs, qs, cfg.chronos2),
    }
    wanted = only or [m for m in LADDER if cfg.models.get(m) and cfg.models[m].enabled]
    unknown = set(wanted) - set(registry) - {ENSEMBLE}
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)} (known: {', '.join(LADDER)})")
    if "chronos2" in wanted and not chronos_available():
        wanted = [m for m in wanted if m != "chronos2"]
        if notes is not None:
            notes.append("chronos2 skipped: install with `uv sync --extra foundation`")
    if cfg.skill.baseline not in wanted:
        wanted = [cfg.skill.baseline, *wanted]  # skill is always measured against it
    return {m: registry[m] for m in wanted if m in registry}


class Band(FiniteModel):
    quantile: float
    log_return: float | None
    price: float | None


class SkillLine(FiniteModel):
    model: str
    skill_vs_naive: float | None
    coverage: float | None
    dir_acc: float | None
    dir_p: float | None
    dm_p: float | None
    dm_p_adj: float | None
    edge: bool
    verdict: str


class HorizonForecast(FiniteModel):
    horizon: int
    model: str
    bands: list[Band]
    p_up: float | None
    skill: SkillLine
    all_models: dict[str, dict[str, float | None]]  # model -> {q0.1.., p_up}: transparency


class ForecastReport(BaseModel):
    symbol: str
    as_of: date
    last_close: float
    unit: str
    horizons: list[HorizonForecast]
    leaderboard: list[LeaderRow]
    eval_start: date
    eval_end: date
    folds: int
    models: list[str]
    notes: list[str]
    generated_at: datetime
    disclaimer: str = DISCLAIMER


def _cache_key(ds: Dataset, cfg: ForecastingConfig, models: list[str]) -> str:
    payload = json.dumps(
        {
            "cfg": cfg.model_dump(mode="json"),
            "models": models,
            "n": len(ds),
            "last": str(ds.dates[-1]),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class CachedEval(BaseModel):
    leaderboard: list[LeaderRow]
    ensemble_weights: dict[int, dict[str, float]]
    eval_start: date
    eval_end: date
    folds: int


def evaluate_cached(
    ds: Dataset,
    factories: dict[str, ModelFactory],
    cfg: ForecastingConfig,
    cache_dir: Path,
    refresh: bool,
) -> CachedEval:
    key = _cache_key(ds, cfg, list(factories))
    path = cache_dir / f"{ds.symbol.replace('=', '_')}_{ds.dates[-1].date()}_{key}.json"
    if path.exists() and not refresh:
        return CachedEval.model_validate_json(path.read_text())
    result: EvalResult = walk_forward(ds, factories, cfg)
    origins = ds.dates[[p for f in result.folds for p in f.origins]]
    cached = CachedEval(
        leaderboard=result.leaderboard,
        ensemble_weights=result.ensemble_weights,
        eval_start=origins[0].date(),
        eval_end=origins[-1].date(),
        folds=len(result.folds),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(cached.model_dump_json(indent=2))
    return cached


def _verdict(
    row: LeaderRow | None,
    h: int,
    best: LeaderRow | None,
    alpha: float,
    n_models: int,
    auto: LeaderRow | None,
) -> str:
    if row is not None and row.edge:
        text = (
            f"{row.model} beats the naive random walk at {h}d out of sample "
            f"(pinball skill {row.skill:+.1%}, DM p={row.dm_p:.3g}, "
            f"Holm-adjusted {row.dm_p_adj:.3g})."
            f" Caveat: it is the best of {n_models} models scored on these same origins, "
            "so its skill is optimistic (winner's curse)"
        )
        if auto is not None and auto.skill is not None:
            p = f"{auto.dm_p_adj:.3g}" if auto.dm_p_adj is not None else "n/a"
            text += (
                f"; picking the leader using earlier folds only (auto_select) scored "
                f"{auto.skill:+.1%} (adjusted p={p})"
            )
        return text + "."
    if best is None:
        return f"No model could be evaluated at {h}d."
    p = f"{best.dm_p_adj:.3g}" if best.dm_p_adj is not None else "n/a"
    return (
        f"No measurable edge at {h}d: no model beats the naive random walk out of sample "
        f"(best: {best.model}, skill {best.skill:+.1%}, adjusted DM p={p} >= {alpha}). "
        "Showing the random-walk band; treat any direction call as noise."
    )


def forecast_dataset(
    ds: Dataset,
    cfg: AppConfig,
    models: list[str] | None = None,
    refresh: bool = False,
    unit: str = "",
    futures: bool = False,
) -> ForecastReport:
    fc = cfg.forecasting
    notes = list(ds.notes)
    factories = model_factories(fc, models, notes)
    cache_dir = (
        fc.leaderboard_dir
        if fc.leaderboard_dir.is_absolute()
        else cfg.config_dir.parent / fc.leaderboard_dir
    )
    ev = evaluate_cached(ds, factories, fc, cache_dir, refresh)

    last = len(ds) - 1
    train, view = ds.until(last), ds.until(last, with_labels=False)
    live: dict[str, dict[int, pd.DataFrame]] = {}
    for name, make in factories.items():
        model: ForecastModel = make()
        model.fit(train)
        live[name] = model.predict(view, [last])
    if ENSEMBLE in [r.model for r in ev.leaderboard]:
        live[ENSEMBLE] = {}
        names = list(factories)
        for h in ds.horizons:
            w = ev.ensemble_weights.get(h) or {m: 1 / len(names) for m in names}
            cols = list(live[names[0]][h].columns)
            arr = np.stack([live[m][h][cols].to_numpy(float) for m in names])
            live[ENSEMBLE][h] = pd.DataFrame(
                combine(arr, np.array([w.get(m, 0.0) for m in names])),
                index=live[names[0]][h].index,
                columns=cols,
            )

    close = float(ds.close.iloc[-1])
    horizons = []
    for h in ds.horizons:
        rows = [r for r in ev.leaderboard if r.horizon == h]
        auto = next((r for r in rows if r.model == AUTO_SELECT), None)
        rows = [r for r in rows if r.model != AUTO_SELECT]  # a procedure, not a live model
        with_edge = [r for r in rows if r.edge]
        best_any = next((r for r in rows if r.model != fc.skill.baseline), None)
        chosen = (
            with_edge[0] if with_edge else next(r for r in rows if r.model == fc.skill.baseline)
        )
        frame = live[chosen.model][h].iloc[0]
        bands = [
            Band(
                quantile=q,
                log_return=float(frame[qcol(q)]),
                price=close * float(np.exp(frame[qcol(q)])) if close > 0 else None,
            )
            for q in fc.quantiles
        ]
        horizons.append(
            HorizonForecast(
                horizon=h,
                model=chosen.model,
                bands=bands,
                p_up=float(frame["p_up"]),
                skill=SkillLine(
                    model=chosen.model,
                    skill_vs_naive=chosen.skill,
                    coverage=chosen.coverage,
                    dir_acc=chosen.dir_acc,
                    dir_p=chosen.dir_p,
                    dm_p=chosen.dm_p,
                    dm_p_adj=chosen.dm_p_adj,
                    edge=chosen.edge,
                    verdict=_verdict(
                        chosen if chosen.edge else None,
                        h,
                        best_any,
                        fc.skill.significance_alpha,
                        n_models=len(rows) - 1,
                        auto=auto,
                    ),
                ),
                all_models={
                    m: {c: _f(live[m][h].iloc[0][c]) for c in live[m][h].columns} for m in live
                },
            )
        )
    if "chronos2" in factories:
        notes.append(
            "Chronos-2 was pretrained on large public time-series corpora that may include "
            "this history; its scores on origins before its release may be contaminated."
        )
    notes.append(
        "auto_select = the leaderboard leader chosen on earlier folds only, re-chosen each "
        "fold: the out-of-sample score of 'use whichever model leads'."
    )
    if futures:
        notes.append(
            "Bands are for the roll-adjusted front month from the last settle; a contract roll "
            "inside the horizon shifts the quoted front-month price by the calendar spread."
        )
    return ForecastReport(
        symbol=ds.symbol,
        as_of=ds.dates[-1].date(),
        last_close=close,
        unit=unit,
        horizons=horizons,
        leaderboard=ev.leaderboard,
        eval_start=ev.eval_start,
        eval_end=ev.eval_end,
        folds=ev.folds,
        models=list(factories) + ([ENSEMBLE] if ENSEMBLE in live else []),
        notes=notes,
        generated_at=datetime.now(UTC),
    )


def _f(x: object) -> float | None:
    v = float(x)  # type: ignore[arg-type]
    return v if np.isfinite(v) else None


def run_forecast(
    svc: DataService,
    symbol: str,
    models: list[str] | None = None,
    as_of: date | None = None,
    refresh: bool = False,
) -> ForecastReport:
    cfg = svc.cfg
    ds = build_dataset(svc, symbol, tuple(cfg.forecasting.horizons), as_of=as_of)
    inst = svc.instrument(symbol)
    futures = inst.roll_rule is not RollRule.NONE
    return forecast_dataset(ds, cfg, models, refresh, unit=inst.unit, futures=futures)
