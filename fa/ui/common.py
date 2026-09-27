"""Shared Streamlit pieces: workspace, sidebar (symbol search, offline mode, key status),
cached service calls and small formatting helpers. Pages call `fa.api` only through here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from fa import api
from fa.data.http import DataUnavailable

DISCLAIMER = "Research tool, paper trading only. Not investment advice."


@st.cache_resource
def get_workspace() -> api.Workspace:
    return api.Workspace.load()


# --- sidebar --------------------------------------------------------------------------


def _label(ws: api.Workspace, symbol: str) -> str:
    inst = ws.cfg.data.instrument(symbol)
    return f"{symbol} · {inst.name}" if inst else f"{symbol} · ad-hoc ticker"


def sidebar() -> None:
    ws = get_workspace()
    with st.sidebar:
        symbols = [i.symbol for i in ws.cfg.data.universe]
        st.session_state.setdefault("symbol", symbols[0])
        st.selectbox(
            "Instrument",
            symbols,
            key="symbol",
            format_func=lambda s: _label(ws, s),
            accept_new_options=True,
            help="Type to search the configured universe, or enter any Yahoo ticker "
            "(e.g. SPY). Ad-hoc tickers get prices, technicals, forecasts and backtests.",
        )
        st.toggle(
            "Offline (cache only)",
            key="offline",
            help="Use the disk cache only; no network calls.",
        )
        keys = api.key_status()
        with st.expander("API keys", expanded=not keys.get("ANTHROPIC_API_KEY", False)):
            for name, ok in keys.items():
                label = (
                    name.removesuffix("_API_KEY")
                    .title()
                    .replace("Eia", "EIA")
                    .replace("Fred", "FRED")
                )
                st.markdown(f"{'✅' if ok else '⚪'} {label} {'' if ok else '(missing)'}")
            st.caption("Keys are read from `.env` and never displayed.")
        st.caption(DISCLAIMER)


def symbol() -> str:
    return str(st.session_state.get("symbol") or get_workspace().cfg.data.universe[0].symbol)


def offline() -> bool:
    return bool(st.session_state.get("offline", False))


# --- cached calls (keyed by their arguments; the workspace is fixed per process) ------


@st.cache_data(ttl=900, show_spinner="Loading prices...")
def price_history(symbol: str, offline: bool) -> Any:
    return api.price_history(get_workspace(), symbol, offline)


@st.cache_data(ttl=900, show_spinner="Building the snapshot...")
def snapshot(symbol: str, offline: bool, curve: bool) -> Any:
    return api.snapshot(get_workspace(), symbol, offline, curve)


@st.cache_data(ttl=3600, show_spinner=False)
def forecast(symbol: str, models: tuple[str, ...], offline: bool) -> Any:
    return api.forecast(get_workspace(), symbol, list(models) or None, offline)


def guarded[T](fn: Callable[[], T], what: str) -> T | None:
    """Run a data call; show a readable error instead of a traceback."""
    try:
        return fn()
    except DataUnavailable as exc:
        st.error(f"{what}: data unavailable. {exc}")
        if offline():
            st.caption("Offline mode is on: only cached data can be used.")
    except ValueError as exc:
        st.error(f"{what}: {exc}")
    return None


# --- formatting -----------------------------------------------------------------------


def pct(x: float | None, signed: bool = True, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def pval(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")


def notes(items: list[str], title: str = "Notes") -> None:
    if items:
        with st.expander(title, expanded=False):
            for n in items:
                st.markdown(f"- {n}")
