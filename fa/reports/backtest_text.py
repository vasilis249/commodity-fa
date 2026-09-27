"""Plain-text rendering of a BacktestReport."""

from __future__ import annotations

from fa.backtest.metrics import Metrics
from fa.backtest.service import BacktestReport


def _p(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _n(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


ROWS: list[tuple[str, str]] = [
    ("total return", "total_return"),
    ("CAGR", "cagr"),
    ("volatility", "vol"),
    ("Sharpe", "sharpe"),
    ("Sortino", "sortino"),
    ("max drawdown", "max_drawdown"),
    ("Calmar", "calmar"),
    ("hit rate", "hit_rate"),
    ("turnover / yr", "turnover"),
    ("exposure", "exposure"),
    ("trades", "trades"),
    ("costs paid", "costs_paid"),
]
PCT = {"total_return", "cagr", "vol", "max_drawdown", "hit_rate", "exposure", "costs_paid"}


def _cell(m: Metrics, key: str) -> str:
    v = getattr(m, key)
    if key == "trades":
        return str(v)
    if key in PCT:
        return _p(v, signed=key in {"total_return", "cagr", "max_drawdown"})
    return _n(v)


def render(r: BacktestReport) -> str:
    s, b = r.strategy_metrics, r.benchmark_metrics
    lines = [
        f"{r.symbol} backtest: {r.strategy} vs buy-and-hold, "
        f"{s.start}..{s.end} ({s.days} sessions)",
        "",
        f"  {'':<16}{r.strategy:>14}{'buy & hold':>14}",
    ]
    lines += [f"  {label:<16}{_cell(s, k):>14}{_cell(b, k):>14}" for label, k in ROWS]
    ci = s.sharpe_ci95
    if ci:
        lines.append(f"  Sharpe 95% CI   [{ci[0]:+.2f}, {ci[1]:+.2f}] (strategy)")
    lines += ["", "NOTES"] + [f"  - {n}" for n in r.notes]
    if r.results_file:
        lines.append(f"  - saved: {r.results_file} (+ equity curves .csv)")
    lines += ["", r.disclaimer]
    return "\n".join(lines)
