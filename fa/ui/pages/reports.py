"""Reports: browse saved agent reports, trace every [T#] citation to its logged tool
result, and run a new report (Anthropic API, hard budget)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from fa import api
from fa.charts import cost_by_agent_figure
from fa.ui import common as c
from fa.ui.pages.forecast import show_forecast

CITE = re.compile(r"\[(T\d+)\]")
RATING_COLOR = {
    "Strong Buy": "green",
    "Buy": "green",
    "Hold": "gray",
    "Sell": "red",
    "Strong Sell": "red",
}


def _fmt(v: float | str | None) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
    return "n/a" if v is None else str(v)


def cited_ids(rep) -> list[str]:
    text = rep.model_dump_json()
    return sorted(set(CITE.findall(text)), key=lambda t: int(t[1:]))


def tool_results(scratchpad: str) -> dict[str, dict]:
    p = Path(scratchpad)
    if not p.exists():
        return {}
    return {
        e["result_id"]: e
        for e in api.run_events(p)
        if e.get("event") == "tool_call" and e.get("result_id")
    }


def _view(rep) -> None:
    color = RATING_COLOR.get(rep.rating, "gray")
    st.subheader(f"{rep.name} ({rep.symbol}), as of {rep.as_of}")
    if rep.lookahead_warning:
        st.warning(rep.lookahead_warning)
    cols = st.columns(4)
    cols[0].markdown(f"### :{color}[{rep.rating}]")
    cols[1].metric("Conviction", c.num(rep.conviction))
    cols[2].metric("Suggested max exposure", c.pct(rep.max_exposure, signed=False, digits=0))
    cols[3].metric(f"Run cost (budget ${rep.cost.budget_usd:.2f})", f"${rep.cost.usd:.3f}")
    st.markdown(rep.thesis)

    with st.expander("How the rating was computed (in code)"):
        st.dataframe(
            pd.DataFrame([x.model_dump() for x in rep.decision.components]), hide_index=True
        )
        st.caption(
            f"Score {rep.decision.score:+.2f}; thresholds in config/agents.yaml. The "
            "synthesizer explains the rating but cannot change it."
        )

    st.markdown(f"**Forecast** ({rep.forecast_eval})")
    pdata = c.guarded(lambda: c.price_history(rep.symbol, True), "Prices (cache)")
    hist = pdata.frame["close"] if pdata is not None else None
    show_forecast(rep.forecast, hist, rep.as_of, f"{rep.symbol}: forecast at {rep.as_of}")

    tabs = st.tabs([s.title for s in rep.sections] + ["Bull vs bear", "Risks", "Validation"])
    for tab, s in zip(tabs, rep.sections, strict=False):
        with tab:
            st.markdown(s.text)
            if s.view:
                v = s.view
                st.caption(f"{s.agent}: {v.stance}, confidence {v.confidence:.2f}")
                for p in v.key_points:
                    st.markdown(f"- {p}")
                if v.red_flags:
                    st.markdown("**Red flags**\n" + "\n".join(f"- {x}" for x in v.red_flags))
                if v.data_gaps:
                    st.markdown("**Data gaps**\n" + "\n".join(f"- {x}" for x in v.data_gaps))
            if s.key_numbers:
                st.dataframe(
                    pd.DataFrame(
                        [(k, _fmt(v)) for k, v in s.key_numbers.items()],
                        columns=["field", "value"],
                    ),
                    hide_index=True,
                )
    n = len(rep.sections)
    with tabs[n]:
        left, right = st.columns(2)
        for col, title, summary, case in (
            (left, "Bull", rep.bull_summary, rep.bull_case),
            (right, "Bear", rep.bear_summary, rep.bear_case),
        ):
            with col:
                st.markdown(f"**{title}**: {summary}")
                if case:
                    for a in case.arguments:
                        st.markdown(f"- {a}")
    with tabs[n + 1]:
        st.markdown(rep.risk_summary)
        for r in rep.risks:
            st.markdown(f"- {r}")
        st.caption(rep.exposure_note)
    with tabs[n + 2]:
        v = rep.validation
        st.markdown(
            f"{v.numeric_violations_fixed} numeric fixes, {len(v.sentences_removed)} sentences "
            f"removed, {len(v.unresolved_validator_issues)} unresolved validator issues."
        )
        for s in v.sentences_removed:
            st.markdown(f"- removed: ~~{s}~~")
        for s in v.unresolved_validator_issues:
            st.markdown(f"- unresolved: {s}")

    st.markdown("**Trace a citation**")
    results = tool_results(rep.scratchpad)
    ids = cited_ids(rep)
    if not results:
        st.caption(f"Scratchpad not found: {rep.scratchpad}")
    elif ids:
        rid = st.selectbox("Result id", ids, key=f"cite_{rep.run_id}")
        e = results.get(rid)
        if e:
            st.caption(f"{e['agent']} called `{e['tool']}` with {e.get('args') or '{}'}")
            st.json(e["result"], expanded=True)
        else:
            st.error(f"{rid} is cited but not in the scratchpad.")

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(cost_by_agent_figure(rep.cost.by_agent), width="stretch")
    with cols[1]:
        st.markdown("**Models**")
        st.dataframe(pd.DataFrame(rep.models.items(), columns=["agent", "model"]), hide_index=True)
        c.notes(rep.data_notes, "Data notes")
        st.caption("Sources: " + ", ".join(rep.data_sources))
    st.caption(rep.disclaimer)


def _downloads(path: str) -> None:
    p = Path(path)
    cols = st.columns(3)
    for col, ext, mime in zip(
        cols,
        ("md", "html", "json"),
        ("text/markdown", "text/html", "application/json"),
        strict=True,
    ):
        f = p.with_suffix(f".{ext}")
        if f.exists():
            col.download_button(f"Download .{ext}", f.read_bytes(), f.name, mime=mime)


def _new_report(ws: api.Workspace) -> None:
    sym, off = c.symbol(), c.offline()
    budget_cap = ws.cfg.models.budget.max_usd_per_run
    has_key = api.key_status().get("ANTHROPIC_API_KEY", False)
    with st.form("new_report"):
        st.markdown(f"New report for **{sym}**")
        as_of = st.date_input(
            "As of (point in time)",
            value=None,
            max_value=date.today(),
            help="Leave empty for the latest data.",
        )
        rounds = st.number_input("Debate rounds", 0, 4, ws.cfg.agents.debate_rounds)
        budget = st.number_input(
            "Budget (USD, hard cap)",
            0.05,
            budget_cap,
            budget_cap,
            0.05,
            help="Capped by models.yaml budget.max_usd_per_run.",
        )
        ok = st.checkbox(
            f"I understand this calls the Anthropic API and may cost up to ${budget:.2f}."
        )
        go = st.form_submit_button("Run report", type="primary", disabled=not has_key)
    if not has_key:
        st.caption("Set ANTHROPIC_API_KEY in `.env` to run reports.")
    if go and not ok:
        st.warning("Tick the cost confirmation first.")
    if go and ok:
        from fa.orchestration.pipeline import PipelineAborted

        with st.status("Running analysts, debate, risk review and validation...") as status:
            try:
                rep, paths = api.run_report(
                    ws,
                    sym,
                    as_of=as_of,
                    offline=off,
                    debate_rounds=int(rounds),
                    budget_usd=float(budget),
                )
            except PipelineAborted as exc:
                status.update(label="Aborted", state="error")
                st.error(
                    f"Aborted: {exc} (spent ${exc.spent_usd:.3f}). Scratchpad: {exc.scratchpad}"
                )
                return
            except Exception as exc:  # show, don't crash the app
                status.update(label="Failed", state="error")
                st.error(f"Report failed: {exc}")
                return
            status.update(label=f"Done: {rep.rating} (${rep.cost.usd:.3f})", state="complete")
        st.session_state["report_path"] = str(paths["json"])


def render() -> None:
    ws = c.get_workspace()
    st.header("Agent reports")
    with st.expander("Run a new report", expanded=False):
        _new_report(ws)
    files = api.list_reports(ws)
    if not files:
        st.info("No saved reports yet. Run one above or with `fa report CL=F`.")
        return
    table = pd.DataFrame([f.model_dump() for f in files])
    st.dataframe(
        table.drop(columns=["path"]),
        hide_index=True,
        column_config={"cost_usd": st.column_config.NumberColumn(format="$%.3f")},
    )
    paths = [f.path for f in files]
    default = st.session_state.get("report_path")
    idx = paths.index(default) if default in paths else 0
    labels = {
        f.path: f"{f.symbol} · {f.as_of} · {f.rating} · {f.generated_at:%Y-%m-%d %H:%M}"
        for f in files
    }
    path = st.selectbox("Report", paths, index=idx, format_func=lambda p: labels[p])
    if path is None:
        return
    rep = api.load_report(path)
    _downloads(path)
    _view(rep)
