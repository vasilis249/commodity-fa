"""Plain-text rendering of a ForecastReport (for `fa forecast`)."""

from __future__ import annotations

from fa.forecasting.service import ForecastReport


def _p(x: float | None, digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def _n(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def _pv(x: float | None) -> str:
    """p-values: 3 significant figures, so 0.0498 never prints as 0.050."""
    return "n/a" if x is None else f"{x:.3g}"


def render(r: ForecastReport) -> str:
    qs = [b.quantile for b in r.horizons[0].bands] if r.horizons else []
    head = "  ".join(f"{'P' + format(q * 100, 'g'):>18}" for q in qs)
    lines = [
        f"{r.symbol} forecast as of {r.as_of}  (last settle {_n(r.last_close)} {r.unit})",
        "",
        f"  {'h':>3}  {'model':<10}{head}  {'P(up)':>6}",
    ]
    for hf in r.horizons:
        cells = "  ".join(f"{_n(b.price):>9} ({_p(b.log_return):>7})" for b in hf.bands)
        pup = _p(hf.p_up, 0, sign=False)
        lines.append(f"  {hf.horizon:>2}d  {hf.model:<10}{cells}  {pup:>6}")
    lines += ["", "MODEL SKILL (out of sample, vs naive random walk)"]
    for hf in r.horizons:
        s = hf.skill
        lines.append(
            f"  {hf.horizon:>2}d  {s.verdict}  [coverage {_p(s.coverage, 0, False)} "
            f"of an 80% band, direction {_p(s.dir_acc, 0, False)} (p={_pv(s.dir_p)})]"
        )
    lines += [
        "",
        f"LEADERBOARD  walk-forward {r.eval_start}..{r.eval_end}, {r.folds} folds "
        "(sorted by pinball loss; skill = 1 - pinball/naive)",
        f"  {'h':>3}  {'model':<10} {'pinball':>8} {'skill':>7} {'MASE':>5} {'cov80':>6} "
        f"{'dir':>5} {'dir p':>6} {'DM p':>6} {'adj p':>6}  edge",
    ]
    for row in r.leaderboard:
        lines.append(
            f"  {row.horizon:>2}d  {row.model:<10} {_n(row.pinball, 5):>8} {_p(row.skill):>7} "
            f"{_n(row.mase):>5} {_p(row.coverage, 0, False):>6} {_p(row.dir_acc, 0, False):>5} "
            f"{_pv(row.dir_p):>6} {_pv(row.dm_p):>6} {_pv(row.dm_p_adj):>6}  "
            f"{'YES' if row.edge else 'no'}"
        )
    if r.notes:
        lines += ["", "NOTES"] + [f"  - {n}" for n in r.notes]
    lines += ["", r.disclaimer]
    return "\n".join(lines)
