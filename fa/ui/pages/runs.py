"""Run history: every agent run's scratchpad (tool calls, LLM calls, checks, cost)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fa import api
from fa.charts import cost_by_agent_figure
from fa.ui import common as c


def cost_by_agent(events: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in events:
        if e.get("event") == "llm_call":
            out[e.get("agent", "?")] = out.get(e.get("agent", "?"), 0.0) + (
                e.get("cost_usd") or 0.0
            )
    return out


def render() -> None:
    ws = c.get_workspace()
    st.header("Run history")
    st.caption(
        f"Scratchpads in `{ws.runs_dir}`: the exact payload of every tool call the "
        "agents saw, every LLM call with its model, prompt version and cost."
    )
    runs = api.list_runs(ws)
    if not runs:
        st.info("No runs yet. Agent reports and agent backtests write one scratchpad each.")
        return
    table = pd.DataFrame([r.model_dump() for r in runs])
    kinds = sorted(table["kind"].unique())
    shown = st.multiselect("Kind", kinds, default=kinds)
    table = table[table["kind"].isin(shown)]
    st.dataframe(
        table.drop(columns=["path"]),
        hide_index=True,
        column_config={"cost_usd": st.column_config.NumberColumn(format="$%.4f")},
    )
    st.metric("Total LLM cost of listed runs", f"${table['cost_usd'].sum():.3f}")
    if table.empty:
        return
    labels = {
        r.path: f"{r.run_id} · {r.kind} · {r.symbol or ''} · {r.started:%Y-%m-%d %H:%M}"
        if r.started
        else r.run_id
        for r in runs
        if r.path in set(table["path"])
    }
    path = st.selectbox("Run", list(labels), format_func=lambda p: labels[p])
    if path is None:
        return
    events = api.run_events(path)
    frame = api.events_frame(events)
    costs = cost_by_agent(events)
    errors = [e for e in events if e.get("event") == "error"]
    for e in errors:
        st.error(f"{e.get('agent') or e.get('stage')}: {e.get('error')}")
    if costs:
        st.plotly_chart(cost_by_agent_figure(costs), width="stretch")
    types = sorted(frame["event"].dropna().unique())
    pick = st.multiselect("Events", types, default=types)
    view = frame[frame["event"].isin(pick)]
    st.dataframe(
        view,
        hide_index=True,
        height=360,
        column_config={"cost_usd": st.column_config.NumberColumn(format="$%.5f")},
    )
    seqs = view["seq"].dropna().astype(int).tolist()
    if seqs:
        seq = st.selectbox("Event detail (seq)", seqs)
        st.json(next(e for e in events if e.get("seq") == seq), expanded=2)
    with open(path, "rb") as f:
        st.download_button("Download scratchpad (.jsonl)", f.read(), file_name=path.split("/")[-1])
