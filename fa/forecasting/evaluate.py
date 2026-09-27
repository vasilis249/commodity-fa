"""Walk-forward evaluation, metrics, significance tests and the ensemble.

Folds: test blocks of `step_days` sessions, expanding training window. Training for a
block starting at row s uses `Dataset.until(s - 1 - embargo)`. There, labels exist only
where t + h <= s - 1 - embargo, so no training label overlaps the test block.
Forecast origins are every `eval_stride`-th session (aligned globally).

Skill: a model has an edge at horizon h only if its pinball loss beats the naive random
walk on the same origins, with a one-sided Diebold-Mariano test (Newey-West variance
for overlapping horizons) whose Holm-adjusted p-value (across the models compared at
that horizon) is below `significance_alpha`. Without the correction, trying 5-6
models per horizon would "find" an edge in a pure random walk far too often.
Directional accuracy is tested with a binomial test on non-overlapping origins only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from fa.analytics.models import FiniteModel
from fa.config import ForecastingConfig
from fa.forecasting.base import ForecastModel, qcol, sort_quantiles
from fa.forecasting.dataset import Dataset

ModelFactory = Callable[[], ForecastModel]
ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class Fold:
    k: int
    train_end: int  # last row of the training view
    test_start: int
    test_end: int
    origins: tuple[int, ...]


def plan_folds(n: int, cfg: ForecastingConfig) -> list[Fold]:
    wf = cfg.walk_forward
    folds: list[Fold] = []
    s = wf.min_train_days
    k = 0
    while s < n:
        e = min(s + wf.step_days, n) - 1
        origins = tuple(p for p in range(s, e + 1) if p % wf.eval_stride == 0)
        if origins:
            folds.append(Fold(k, s - 1 - wf.embargo_days, s, e, origins))
            k += 1
        s += wf.step_days
    return folds[-wf.max_folds :]


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    d = y - pred
    return np.maximum(q * d, (q - 1) * d)


def mean_pinball(y: np.ndarray, frame: pd.DataFrame, quantiles: Sequence[float]) -> np.ndarray:
    """Per-origin pinball loss averaged over quantiles."""
    losses = [pinball(y, frame[qcol(q)].to_numpy(), q) for q in quantiles]
    return np.mean(losses, axis=0)


def diebold_mariano(d: np.ndarray, lag: int) -> float:
    """One-sided p-value that mean(d) < 0 (model loss below baseline), NW variance."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return float("nan")
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return float("nan")
    return float(stats.norm.cdf(d.mean() / math.sqrt(var / n)))


def non_overlapping(dates: pd.DatetimeIndex, positions: np.ndarray, h: int) -> np.ndarray:
    """Greedy subset of origin positions spaced at least h sessions apart."""
    keep, last = [], -(10**9)
    for i, p in enumerate(positions):
        if p - last >= h:
            keep.append(i)
            last = p
    return np.asarray(keep, dtype=int)


class LeaderRow(FiniteModel):
    model: str
    horizon: int
    n: int
    n_independent: int
    mae: float | None
    mase: float | None  # MAE / naive MAE (same origins)
    pinball: float | None
    skill: float | None  # 1 - pinball / naive pinball
    coverage: float | None  # share inside the outermost quantiles (target in config)
    dir_acc: float | None
    dir_p: float | None  # binomial, non-overlapping origins
    brier: float | None
    dm_p: float | None  # Diebold-Mariano vs naive, one-sided
    dm_p_adj: float | None  # Holm-adjusted across models at this horizon
    edge: bool


@dataclass
class EvalResult:
    predictions: dict[str, dict[int, pd.DataFrame]]  # model -> horizon -> rows by origin
    realized: dict[int, pd.Series]
    positions: dict[int, np.ndarray]
    leaderboard: list[LeaderRow]
    ensemble_weights: dict[int, dict[str, float]]  # for the final forecast
    folds: list[Fold] = field(default_factory=list)


def walk_forward(
    ds: Dataset, factories: dict[str, ModelFactory], cfg: ForecastingConfig
) -> EvalResult:
    folds = plan_folds(len(ds), cfg)
    if not folds:
        raise ValueError(f"not enough history: {len(ds)} rows < min_train_days + 1")
    preds: dict[str, dict[int, list[pd.DataFrame]]] = {
        m: {h: [] for h in ds.horizons} for m in factories
    }
    fold_of: list[int] = []
    fitted: dict[str, ForecastModel] = {}
    for i, fold in enumerate(folds):
        train = ds.until(fold.train_end)
        view = ds.until(fold.test_end, with_labels=False)
        refit = i % cfg.walk_forward.refit_every == 0
        for name, make in factories.items():
            if refit or name not in fitted:
                fitted[name] = make()
                fitted[name].fit(train)  # older folds' fits only ever saw older data
            out = fitted[name].predict(view, fold.origins)
            for h in ds.horizons:
                preds[name][h].append(out[h])
        fold_of.extend([fold.k] * len(fold.origins))

    stacked = {m: {h: pd.concat(v) for h, v in hs.items()} for m, hs in preds.items()}
    origin_pos = np.concatenate([np.asarray(f.origins) for f in folds])
    fold_ids = np.asarray(fold_of)
    realized = {h: ds.label(h).iloc[origin_pos] for h in ds.horizons}

    weights: dict[int, dict[str, float]] = {}
    if (
        cfg.models.get(ENSEMBLE, None) is not None
        and cfg.models[ENSEMBLE].enabled
        and len(factories) > 1
    ):
        ens, weights = _ensemble(stacked, realized, fold_ids, cfg)
        stacked[ENSEMBLE] = ens

    positions = {h: origin_pos for h in ds.horizons}
    board = _leaderboard(stacked, realized, positions, ds, cfg)
    return EvalResult(stacked, realized, positions, board, weights, folds)


def _ensemble(
    preds: dict[str, dict[int, pd.DataFrame]],
    realized: dict[int, pd.Series],
    fold_ids: np.ndarray,
    cfg: ForecastingConfig,
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, float]]]:
    """Weights per horizon = inverse mean pinball on earlier folds only (causal)."""
    qs = cfg.quantiles
    names = list(preds)
    out: dict[int, pd.DataFrame] = {}
    final: dict[int, dict[str, float]] = {}
    for h, y in realized.items():
        yv = y.to_numpy()
        losses = {m: mean_pinball(yv, preds[m][h], qs) for m in names}
        cols = list(preds[names[0]][h].columns)
        arr = np.stack([preds[m][h][cols].to_numpy(dtype=float) for m in names])  # (M, N, C)
        blended = np.full(arr.shape[1:], np.nan)
        for k in np.unique(fold_ids):
            now = fold_ids == k
            w = _weights(losses, fold_ids, int(k), cfg)
            blended[now] = combine(arr[:, now, :], np.array([w[m] for m in names]))
        frame = pd.DataFrame(blended, index=preds[names[0]][h].index, columns=cols)
        out[h] = sort_quantiles(frame, qs)
        final[h] = _weights(losses, fold_ids, int(fold_ids.max()) + 1, cfg)
    return out, final


def combine(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average over axis 0, renormalized over models with a forecast."""
    valid = np.isfinite(arr)
    w = weights[:, None, None] * valid
    total = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, np.nansum(np.where(valid, arr, 0.0) * w, axis=0) / total, np.nan)


def _weights(
    losses: dict[str, np.ndarray], fold_ids: np.ndarray, k: int, cfg: ForecastingConfig
) -> dict[str, float]:
    ec = cfg.ensemble
    earlier = np.unique(fold_ids[fold_ids < k])[-ec.window_folds :]
    past = np.isin(fold_ids, earlier)
    names = list(losses)
    if len(earlier) < ec.min_folds:
        return {m: 1 / len(names) for m in names}
    mean_loss = {m: np.nanmean(losses[m][past]) for m in names}
    inv = {m: 1 / v if np.isfinite(v) and v > 0 else 0.0 for m, v in mean_loss.items()}
    total = sum(inv.values())
    return (
        {m: v / total for m, v in inv.items()} if total > 0 else {m: 1 / len(names) for m in names}
    )


def _leaderboard(
    preds: dict[str, dict[int, pd.DataFrame]],
    realized: dict[int, pd.Series],
    positions: dict[int, np.ndarray],
    ds: Dataset,
    cfg: ForecastingConfig,
) -> list[LeaderRow]:
    qs = cfg.quantiles
    base = cfg.skill.baseline
    rows: list[LeaderRow] = []
    for h, y in realized.items():
        yv = y.to_numpy()
        ok_base = np.isfinite(yv) & preds[base][h][qcol(0.5)].notna().to_numpy()
        lag = max(0, math.ceil(h / cfg.walk_forward.eval_stride) - 1)
        base_loss = mean_pinball(yv, preds[base][h], qs)
        base_mae = np.abs(yv - preds[base][h][qcol(0.5)].to_numpy())
        for name, by_h in preds.items():
            f = by_h[h]
            ok = ok_base & f[qcol(0.5)].notna().to_numpy()
            n = int(ok.sum())
            if n == 0:
                continue
            loss = mean_pinball(yv, f, qs)
            med = f[qcol(0.5)].to_numpy()
            mae = float(np.mean(np.abs(yv - med)[ok]))
            lo, hi = f[qcol(min(qs))].to_numpy(), f[qcol(max(qs))].to_numpy()
            cover = float(np.mean(((yv >= lo) & (yv <= hi))[ok]))
            pup = f["p_up"].to_numpy()
            up = yv > 0
            brier = float(np.nanmean(((pup - up) ** 2)[ok]))
            indep = non_overlapping(ds.dates, positions[h][ok], h)
            hits = ((pup >= 0.5) == up)[ok][indep]
            nonzero = (yv[ok][indep] != 0) & np.isfinite(pup[ok][indep])
            k_hit, n_dir = int(hits[nonzero].sum()), int(nonzero.sum())
            dir_p = (
                stats.binomtest(k_hit, n_dir, 0.5, alternative="greater").pvalue
                if n_dir
                else float("nan")
            )
            pin = float(np.mean(loss[ok]))
            base_pin = float(np.mean(base_loss[ok]))
            dm_p = float("nan") if name == base else diebold_mariano(loss[ok] - base_loss[ok], lag)
            skill = 1 - pin / base_pin if base_pin > 0 else float("nan")
            rows.append(
                LeaderRow(
                    model=name,
                    horizon=h,
                    n=n,
                    n_independent=n_dir,
                    mae=mae,
                    mase=mae / float(np.mean(base_mae[ok])) if np.mean(base_mae[ok]) > 0 else None,
                    pinball=pin,
                    skill=skill,
                    coverage=cover,
                    dir_acc=k_hit / n_dir if n_dir else None,
                    dir_p=float(dir_p),
                    brier=brier,
                    dm_p=dm_p,
                    dm_p_adj=None,
                    edge=False,
                )
            )
    alpha = cfg.skill.significance_alpha
    for h in realized:
        cands = [r for r in rows if r.horizon == h and r.dm_p is not None]
        for r, adj in zip(cands, holm([r.dm_p for r in cands]), strict=True):  # type: ignore[misc]
            r.dm_p_adj = adj
            r.edge = bool(r.skill is not None and r.skill > 0 and adj < alpha)
    return sorted(rows, key=lambda r: (r.horizon, r.pinball if r.pinball is not None else np.inf))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()
