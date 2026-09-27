"""Market: price chart (quoted front month, roll-adjusted close, SMAs, rolls) and the
numeric point-in-time snapshot. No LLM involved."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fa.charts import price_figure
from fa.ui import common as c


def _metrics(snap) -> None:
    p, t, r = snap.price, snap.technicals, snap.risk
    cols = st.columns(6)
    cols[0].metric(f"Last ({snap.unit})", c.num(p.last_close), c.pct(p.change_1d))
    cols[1].metric("5 days", c.pct(p.change_5d))
    cols[2].metric("20 days", c.pct(p.change_20d))
    cols[3].metric("From 52w high", c.pct(p.pct_from_52w_high))
    cols[4].metric("Vol 20d (ann.)", c.pct(r.vol_20d, signed=False))
    cols[5].metric("Trend", t.trend, f"vol {t.vol_regime}", delta_color="off")


def _technicals(snap) -> None:
    t = snap.technicals
    rows = {
        "SMA 20 / 50 / 200": f"{c.num(t.sma20)} / {c.num(t.sma50)} / {c.num(t.sma200)}",
        "RSI 14": c.num(t.rsi14, 1),
        "MACD / signal / hist": f"{c.num(t.macd)} / {c.num(t.macd_signal)} / {c.num(t.macd_hist)}",
        "Bollinger %b": c.num(t.bb_pct_b),
        "ATR 14 (% of price)": f"{c.num(t.atr14)} ({c.pct(t.atr_pct, signed=False)})",
        "Vol percentile (3y)": c.num(t.vol_percentile_3y, 0),
        "Support": ", ".join(c.num(lv.price) for lv in t.support) or "none",
        "Resistance": ", ".join(c.num(lv.price) for lv in t.resistance) or "none",
    }
    st.table(pd.DataFrame(rows.items(), columns=["indicator", "value"]).set_index("indicator"))


def _risk(snap) -> None:
    r = snap.risk
    rows = {
        "Volatility 1y": c.pct(r.vol_1y, signed=False),
        "Return 1y": c.pct(r.return_1y),
        "Sharpe / Sortino 1y": f"{c.num(r.sharpe_1y)} / {c.num(r.sortino_1y)}",
        "Max drawdown 1y": c.pct(r.max_drawdown_1y),
        "Drawdown from 1y high": c.pct(r.drawdown_from_1y_high),
        "VaR / CVaR 95% (1 day)": f"{c.pct(r.var95_1d)} / {c.pct(r.cvar95_1d)}",
        "Suggested max exposure": c.pct(r.position.fraction, signed=False, digits=0),
    }
    st.table(pd.DataFrame(rows.items(), columns=["metric", "value"]).set_index("metric"))
    st.caption("Exposure rule: " + "; ".join(r.position.reasons))


def _fundamentals(snap) -> None:
    if snap.inventories:
        st.markdown("**Weekly fundamentals** (point in time: as released)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "series": i.name,
                        "period": i.period,
                        "released": i.released_at,
                        "value": i.value,
                        "units": i.units,
                        "w/w": i.wow,
                        "vs 5y avg": c.pct(i.dev_pct),
                        "5y band position": c.num(i.band_pos),
                    }
                    for i in snap.inventories
                ]
            ),
            hide_index=True,
        )
    if snap.cot:
        k = snap.cot
        st.markdown(
            f"**CFTC positioning** (as of {k.report_date}, released {k.released_at:%Y-%m-%d}): "
            f"managed money net {k.mm_net:,.0f} contracts, {c.pct(k.mm_net_pct_oi)} of open "
            f"interest, 3y percentile {c.num(k.mm_net_pctile_3y, 0)}."
        )
    if snap.crack:
        k = snap.crack
        st.markdown(
            f"**3-2-1 crack spread** ({k.date}): {c.num(k.value)} USD/bbl, 1y percentile "
            f"{c.num(k.pctile_1y, 0)}"
            + (" (a leg is inside a roll window)." if k.roll_affected else ".")
        )
    if snap.seasonality:
        s = snap.seasonality.stats
        st.markdown(
            f"**Seasonality ({snap.seasonality.month_name})**: mean log return "
            f"{c.pct(s.mean)}, positive in {c.pct(s.hit_rate, signed=False, digits=0)} of "
            f"{s.years} years (t = {c.num(s.t_stat)})."
        )
    if not (snap.inventories or snap.cot or snap.crack):
        st.info("No fundamental series are configured for this instrument.")


def _curve(snap) -> None:
    if not snap.curve:
        st.info("No futures curve (offline, not a futures contract, or unavailable).")
        return
    q = snap.curve.contracts
    fig = go.Figure(
        go.Scatter(x=[x.contract for x in q], y=[x.settle for x in q], mode="lines+markers")
    )
    m = snap.curve.metrics
    fig.update_layout(
        title=f"Term structure: {m.structure} (M1-M2 {c.pct(m.m1_m2_pct)}, "
        f"annualized roll yield {c.pct(m.roll_yield_ann)})",
        height=320,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")


def render() -> None:
    sym, off = c.symbol(), c.offline()
    ws = c.get_workspace()
    inst = ws.cfg.data.instrument(sym)
    st.header(f"{sym} · {inst.name if inst else 'ad-hoc ticker'}")
    pdata = c.guarded(lambda: c.price_history(sym, off), "Prices")
    if pdata is None:
        return
    years = st.segmented_control("Window", [1, 2, 5, 10], default=2, format_func=lambda y: f"{y}y")
    st.plotly_chart(
        price_figure(
            pdata.frame, f"{sym} ({pdata.instrument.unit})", pdata.rolls, float(years or 2)
        ),
        width="stretch",
    )
    issues = [i for i in pdata.quality.issues if i.count]
    if issues:
        with st.expander(f"Data quality: {len(issues)} checks flagged"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "check": i.check,
                            "severity": str(i.severity),
                            "count": i.count,
                            "detail": i.detail,
                        }
                        for i in issues
                    ]
                ),
                hide_index=True,
            )

    curve = st.checkbox("Fetch the futures curve", value=not off and inst is not None)
    snap = c.guarded(lambda: c.snapshot(sym, off, curve), "Snapshot")
    if snap is None:
        return
    st.caption(
        f"Point-in-time snapshot as of {snap.as_of} "
        f"(decision time {snap.decision_time:%Y-%m-%d %H:%M} UTC)"
    )
    _metrics(snap)
    tabs = st.tabs(["Technicals", "Risk", "Fundamentals", "Curve", "Sources"])
    with tabs[0]:
        _technicals(snap)
    with tabs[1]:
        _risk(snap)
    with tabs[2]:
        _fundamentals(snap)
    with tabs[3]:
        _curve(snap)
    with tabs[4]:
        st.dataframe(pd.DataFrame([s.model_dump() for s in snap.sources]), hide_index=True)
        c.notes(snap.notes, "Snapshot notes")
