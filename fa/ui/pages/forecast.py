"""Forecast: fan chart with measured skill, per-horizon table and the leaderboard.

Guardrail: a forecast is never shown without its skill metrics and uncertainty band. The
fan chart embeds the skill line of every horizon, and the table beside it repeats it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fa.charts import fan_chart, leaderboard_figure
from fa.ui import common as c


def horizon_table(horizons) -> pd.DataFrame:
    rows = []
    for h in horizons:
        bands = {f"P{b.quantile * 100:g}": b.price for b in h.bands}
        rows.append(
            {
                "horizon": f"{h.horizon}d",
                "model": h.model,
                **{k: c.num(v) for k, v in bands.items()},
                "P(up)": c.pct(h.p_up, signed=False, digits=0),
                "skill vs random walk": c.pct(h.skill.skill_vs_naive),
                "band coverage": c.pct(h.skill.coverage, signed=False, digits=0),
                "direction acc.": c.pct(h.skill.dir_acc, signed=False, digits=0),
                "DM p (Holm)": c.pval(h.skill.dm_p_adj),
                "verdict": h.skill.verdict,
            }
        )
    return pd.DataFrame(rows)


def show_forecast(horizons, history: pd.Series | None, as_of, title: str) -> None:
    """Fan chart + skill table; used by the forecast and report pages."""
    if not any(h.skill.edge for h in horizons):
        st.info(
            "**No measurable edge**: no model beat the random walk out of sample at any "
            "horizon. The band shown is the random walk's."
        )
    if history is not None and len(history):
        st.plotly_chart(fan_chart(history, horizons, as_of, title), width="stretch")
    st.dataframe(horizon_table(horizons), hide_index=True)


def render() -> None:
    sym, off = c.symbol(), c.offline()
    ws = c.get_workspace()
    fc = ws.cfg.forecasting
    st.header(f"{sym} · probabilistic forecast")
    enabled = [m for m, spec in fc.models.items() if spec.enabled]
    with st.form("forecast"):
        models = st.multiselect(
            "Models to evaluate (walk-forward, out of sample)",
            list(fc.models),
            default=enabled,
            help="The first run evaluates every model walk-forward (about a minute on a "
            "laptop); the leaderboard is then cached on disk.",
        )
        go = st.form_submit_button("Run forecast", type="primary")
    key = (sym, tuple(models), off)
    if go:
        st.session_state["forecast_key"] = key
    if st.session_state.get("forecast_key") != key:
        st.caption(
            f"Horizons {fc.horizons} trading days, quantiles {fc.quantiles}. "
            "Press **Run forecast**."
        )
        return
    with st.spinner("Walk-forward evaluation and live forecast..."):
        rep = c.guarded(lambda: c.forecast(sym, tuple(models), off), "Forecast")
    if rep is None:
        return
    pdata = c.guarded(lambda: c.price_history(sym, off), "Prices")
    hist = pdata.frame["close"] if pdata is not None else None
    st.caption(
        f"Origin {rep.as_of}, last close {c.num(rep.last_close)} {rep.unit}. Evaluated "
        f"walk-forward {rep.eval_start}..{rep.eval_end} ({rep.folds} folds)."
    )
    show_forecast(rep.horizons, hist, rep.as_of, f"{sym}: forecast bands from the last close")
    c.notes(rep.notes, "Forecast notes")

    st.subheader("Leaderboard")
    lb = pd.DataFrame([r.model_dump() for r in rep.leaderboard])
    tabs = st.tabs([f"{h}d" for h in fc.horizons])
    for tab, h in zip(tabs, fc.horizons, strict=True):
        with tab:
            st.plotly_chart(leaderboard_figure(lb, h), width="stretch")
            st.dataframe(
                lb[lb["horizon"] == h].drop(columns=["horizon"]),
                hide_index=True,
                column_config={
                    "skill": st.column_config.NumberColumn(format="percent"),
                    "coverage": st.column_config.NumberColumn(format="percent"),
                    "dir_acc": st.column_config.NumberColumn(format="percent"),
                },
            )
    st.caption(rep.disclaimer)
