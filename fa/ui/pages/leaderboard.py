"""Leaderboards: every cached walk-forward evaluation, across instruments."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fa import api
from fa.charts import leaderboard_figure
from fa.ui import common as c


def summary(files: list[api.LeaderboardFile]) -> pd.DataFrame:
    """Best non-baseline model per (evaluation, horizon), with its edge verdict."""
    rows = []
    for f in files:
        lb = api.load_leaderboard(f.path)
        for h, g in lb.groupby("horizon"):
            g = g[~g["model"].isin(["naive", "auto_select"])].sort_values("skill", ascending=False)
            if g.empty:
                continue
            best = g.iloc[0]
            rows.append(
                {
                    "symbol": f.symbol,
                    "data to": f.last_date,
                    "horizon": int(h),
                    "best model": best["model"],
                    "skill": best["skill"],
                    "DM p (Holm)": best["dm_p_adj"],
                    "edge": bool(g["edge"].any()),
                    "folds": f.folds,
                }
            )
    return pd.DataFrame(rows)


def render() -> None:
    ws = c.get_workspace()
    st.header("Leaderboards")
    st.caption(
        "Out-of-sample pinball skill vs the random walk from cached walk-forward "
        "evaluations. An edge needs positive skill and a Holm-adjusted Diebold-Mariano "
        f"p < {ws.cfg.forecasting.skill.significance_alpha}."
    )
    files = api.list_leaderboards(ws)
    if not files:
        st.info("No evaluations yet. Run a forecast (Forecast page or `fa forecast CL=F`).")
        return
    s = summary(files)
    st.dataframe(
        s,
        hide_index=True,
        column_config={"skill": st.column_config.NumberColumn(format="percent")},
    )
    labels = {
        f.path: f"{f.symbol} · data to {f.last_date} · {len(f.models)} models · {f.folds} folds"
        for f in files
    }
    path = st.selectbox("Evaluation", list(labels), format_func=lambda p: labels[p])
    if path is None:
        return
    lb = api.load_leaderboard(path)
    for h in sorted(lb["horizon"].unique()):
        st.plotly_chart(leaderboard_figure(lb, int(h)), width="stretch")
    with st.expander("Full table"):
        st.dataframe(lb, hide_index=True)
