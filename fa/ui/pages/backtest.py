"""Backtest: run a paper strategy vs buy-and-hold, and browse saved backtests."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fa import api
from fa.backtest.agent_signal import BacktestCostWarning
from fa.charts import equity_figure
from fa.reports.backtest_text import ROWS, cell
from fa.ui import common as c

STRATEGIES = {
    "sma": "SMA crossover",
    "forecast": "Forecast signal (walk-forward P(up))",
    "agent": "LLM agents on anonymized data",
}


def metrics_table(rep) -> pd.DataFrame:
    s, b = rep.strategy_metrics, rep.benchmark_metrics
    return pd.DataFrame(
        {
            "metric": [label for label, _ in ROWS],
            rep.strategy: [cell(s, k) for _, k in ROWS],
            "buy & hold": [cell(b, k) for _, k in ROWS],
        }
    ).set_index("metric")


def show_backtest(rep, curves: pd.DataFrame | None, key: str) -> None:
    s = rep.strategy_metrics
    first = rep.notes[0] if rep.notes else ""
    (st.info if first.startswith("No measurable edge") else st.success)(first)
    cols = st.columns([2, 3])
    with cols[0]:
        st.caption(f"{rep.symbol} · {rep.strategy} · {s.start}..{s.end} ({s.days} sessions)")
        st.table(metrics_table(rep))
        ci = s.sharpe_ci95
        st.caption(
            (f"Sharpe 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]. " if ci else "")
            + f"One-sided p (beats buy & hold): {c.pval(rep.edge_p_value)}."
        )
    with cols[1]:
        if curves is not None:
            st.plotly_chart(equity_figure(curves, rep.strategy), width="stretch", key=f"eq_{key}")
    c.notes(rep.notes[1:], "Notes (costs, timing, accounting)")
    with st.expander("Parameters"):
        st.json(rep.params)
    st.caption(rep.disclaimer)


def _run(ws: api.Workspace) -> None:
    sym, off = c.symbol(), c.offline()
    bt = ws.cfg.backtest
    strategy = st.radio(
        "Strategy", list(STRATEGIES), format_func=lambda s: STRATEGIES[s], horizontal=True
    )
    risk = ws.cfg.risk.backtest
    st.caption(
        f"Long/short: {bt.long_short} · vol target: {bt.vol_target} · costs "
        f"{risk.cost_bps_per_side:g} + {risk.slippage_bps_per_side:g} bps per side, roll "
        f"{risk.roll_cost_bps:g} bps (config/backtest.yaml, config/risk.yaml)."
    )
    confirm = False
    if strategy == "sma":
        st.caption(f"Long when SMA {bt.sma.fast} > SMA {bt.sma.slow}.")
    elif strategy == "forecast":
        f = bt.forecast
        st.caption(
            f"{f.model} P(up) at {f.horizon}d: long ≥ {f.up_threshold}, "
            f"{'short' if bt.long_short else 'flat'} ≤ {f.down_threshold}. Thresholds are "
            "fixed in config, not tuned here."
        )
    else:
        a = bt.agent
        est = a.max_decisions * a.est_usd_per_decision
        cap = ws.cfg.models.budget.max_usd_backtest
        st.caption(
            f"Up to {a.max_decisions} decisions every {a.rebalance_every} sessions, "
            f"estimated ${est:.2f}; hard cap ${cap:.2f} (models.yaml). Agents see rebased "
            "prices and ratios only: no names, dates, units or news."
        )
        confirm = st.checkbox(f"I accept LLM costs up to ${cap:.2f}.")
        if not api.key_status().get("ANTHROPIC_API_KEY", False):
            st.warning("Set ANTHROPIC_API_KEY in `.env` to run agent backtests.")
            return
    if st.button("Run backtest", type="primary"):
        if strategy == "agent" and not confirm:
            st.warning("Tick the cost confirmation first.")
            return
        with st.spinner(f"Backtesting {sym} ({STRATEGIES[strategy]})..."):
            try:
                rep = c.guarded(
                    lambda: api.run_backtest(ws, sym, strategy, off, confirm), "Backtest"
                )
            except BacktestCostWarning as exc:
                st.warning(str(exc))
                return
        if rep is None:
            return
        st.session_state["last_backtest"] = rep.results_file
    last = st.session_state.get("last_backtest")
    if last:
        rep, curves = api.load_backtest(last)
        st.divider()
        show_backtest(rep, curves, "last")


def render() -> None:
    ws = c.get_workspace()
    st.header(f"{c.symbol()} · paper backtest")
    run_tab, saved_tab = st.tabs(["Run", "Saved backtests"])
    with run_tab:
        _run(ws)
    with saved_tab:
        files = api.list_backtests(ws)
        if not files:
            st.info("No saved backtests yet.")
            return
        st.dataframe(
            pd.DataFrame([f.model_dump() for f in files]).drop(columns=["path"]),
            hide_index=True,
            column_config={
                "cagr": st.column_config.NumberColumn(format="percent"),
                "benchmark_cagr": st.column_config.NumberColumn(format="percent"),
            },
        )
        labels = {f.path: f"{f.symbol} · {f.strategy} · {f.start}..{f.end}" for f in files}
        path = st.selectbox("Backtest", list(labels), format_func=lambda p: labels[p])
        if path is None:
            return
        rep, curves = api.load_backtest(path)
        show_backtest(rep, curves, "saved")
