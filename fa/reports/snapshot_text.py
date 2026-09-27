"""Plain-text rendering of a numeric Snapshot (for `fa analyze --no-llm`)."""

from __future__ import annotations

from fa.analytics.snapshot import Snapshot


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"


def _pct_abs(x: float | None, digits: int = 1) -> str:
    """Unsigned percentage for magnitudes (volatility, VaR)."""
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def render(s: Snapshot) -> str:
    p, t, r = s.price, s.technicals, s.risk
    lines = [
        f"{s.symbol} — {s.name} [{s.unit}]",
        f"as of {s.as_of} settlement ({s.decision_time:%Y-%m-%d %H:%M} UTC)",
        "",
        "PRICE",
        f"  last {_num(p.last_close)}   1d {_pct(p.change_1d)}   5d {_pct(p.change_5d)}   "
        f"20d {_pct(p.change_20d)}   60d {_pct(p.change_60d)}   YTD {_pct(p.change_ytd)}",
        f"  52w range {_num(p.low_52w)} - {_num(p.high_52w)}   "
        f"from 52w high {_pct(p.pct_from_52w_high)}",
        "",
        "TECHNICALS (roll-adjusted series)",
        f"  trend {t.trend}   vol regime {t.vol_regime} (3y pctile {_num(t.vol_percentile_3y, 0)})",
        f"  SMA20 {_num(t.sma20)}  SMA50 {_num(t.sma50)}  SMA200 {_num(t.sma200)}  "
        f"EMA20 {_num(t.ema20)}",
        f"  RSI14 {_num(t.rsi14, 1)}   MACD {_num(t.macd)} / signal {_num(t.macd_signal)} "
        f"/ hist {_num(t.macd_hist)}",
        f"  Bollinger {_num(t.bb_lower)} - {_num(t.bb_upper)} (%b {_num(t.bb_pct_b)})   "
        f"ATR14 {_num(t.atr14)} ({_pct_abs(t.atr_pct)})",
        "  support    "
        + (", ".join(f"{_num(lv.price)} (x{lv.touches})" for lv in t.support) or "none"),
        "  resistance "
        + (", ".join(f"{_num(lv.price)} (x{lv.touches})" for lv in t.resistance) or "none"),
        "",
        "RISK (1y, daily)",
        f"  vol 20d {_pct_abs(r.vol_20d)}  vol 1y {_pct_abs(r.vol_1y)}  "
        f"return 1y {_pct(r.return_1y)}  "
        f"Sharpe {_num(r.sharpe_1y)}  Sortino {_num(r.sortino_1y)}",
        f"  max DD 1y {_pct(r.max_drawdown_1y)}  from 1y high {_pct(r.drawdown_from_1y_high)}  "
        f"VaR95 1d {_pct_abs(r.var95_1d)}  CVaR95 1d {_pct_abs(r.cvar95_1d)}",
        f"  suggested max exposure {r.position.fraction:.0%} of equity "
        f"(binding: {r.position.binding}; {'; '.join(r.position.reasons)})",
    ]
    if s.seasonality:
        m = s.seasonality.stats
        lines += [
            "",
            f"SEASONALITY ({s.seasonality.month_name}, {m.years} years "
            f"{m.first_year}-{m.last_year})",
            f"  mean {_pct(m.mean)}  median {_pct(m.median)}  up-years {_pct(m.hit_rate, 0)}  "
            f"t-stat {_num(m.t_stat)}",
        ]
    if s.inventories:
        lines += ["", "INVENTORIES (EIA, as released by the decision time)"]
        for inv in s.inventories:
            lines.append(
                f"  {inv.name:<22} {inv.period}  {_num(inv.value, 0)} {inv.units or ''}  "
                f"w/w {_num(inv.wow, 0)} (vs seasonal {_num(inv.wow_vs_seasonal, 0)})  "
                f"vs 5y avg {_pct(inv.dev_pct)}  band pos {_num(inv.band_pos)}"
            )
    if s.cot:
        c = s.cot
        lines += [
            "",
            f"POSITIONING (CFTC, report {c.report_date}, released {c.released_at:%Y-%m-%d})",
            f"  managed money net {_num(c.mm_net, 0)} ({_pct(c.mm_net_pct_oi)} of OI)  "
            f"1w change {_num(c.mm_net_change_1w, 0)}  3y percentile {_num(c.mm_net_pctile_3y, 0)}",
        ]
    if s.crack:
        lines += [
            "",
            f"3-2-1 CRACK  {_num(s.crack.value)} USD/bbl on {s.crack.date}  "
            f"(1y percentile {_num(s.crack.pctile_1y, 0)})",
        ]
    if s.curve:
        cm = s.curve.metrics
        quotes = "  ".join(f"{q.contract} {_num(q.settle)}" for q in s.curve.contracts)
        lines += [
            "",
            f"CURVE ({cm.structure})",
            f"  {quotes}",
            f"  M1-M2 {_num(cm.m1_m2_spread)} ({_pct(cm.m1_m2_pct, 2)})  "
            f"roll yield {_pct(cm.roll_yield_ann)}/yr  "
            f"front-to-back {_pct(cm.front_to_back_ann)}/yr",
        ]
    if s.notes:
        lines += ["", "NOTES"] + [f"  - {n}" for n in s.notes]
    lines += [
        "",
        "SOURCES " + ", ".join(sorted({f"{x.provider}:{x.key}" for x in s.sources})),
        s.disclaimer,
    ]
    return "\n".join(lines)
