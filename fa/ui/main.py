"""Streamlit entry point: `uv run streamlit run app/main.py`."""

from __future__ import annotations

import streamlit as st

from fa.ui import common
from fa.ui.pages import backtest, forecast, leaderboard, market, reports, runs

PAGES = [
    st.Page(
        market.render,
        title="Market",
        icon=":material/candlestick_chart:",
        default=True,  # served at "/"
    ),
    st.Page(forecast.render, title="Forecast", icon=":material/insights:", url_path="forecast"),
    st.Page(
        leaderboard.render,
        title="Leaderboards",
        icon=":material/leaderboard:",
        url_path="leaderboards",
    ),
    st.Page(reports.render, title="Reports", icon=":material/description:", url_path="reports"),
    st.Page(backtest.render, title="Backtest", icon=":material/ssid_chart:", url_path="backtest"),
    st.Page(runs.render, title="Run history", icon=":material/history:", url_path="runs"),
]


def main() -> None:
    st.set_page_config(page_title="commodity-fa", page_icon=":material/oil_barrel:", layout="wide")
    page = st.navigation(PAGES)
    common.sidebar()
    page.run()
