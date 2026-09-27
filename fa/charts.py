"""Plotly figure builders for the dashboard. Pure functions of data, no Streamlit, so
they are unit-tested and reusable by any front end.

Guardrail: a forecast figure is never built without its skill line; `fan_chart` takes
the whole `HorizonForecast` (bands + measured skill + verdict), not bare quantiles.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fa.analytics.indicators import sma
from fa.data.rolls import RollEvent
from fa.forecasting.service import HorizonForecast

UP, DOWN = "#2e7d32", "#c62828"
STRATEGY, BENCH = "#1f6feb", "#8b949e"
BAND = "rgba(31,111,235,{a})"


def _pct(x: float | None, signed: bool = True, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _p(x: float | None) -> str:
    return "n/a" if x is None else ("<0.001" if x < 0.001 else f"{x:.3f}")


# --- price ----------------------------------------------------------------------------


def price_figure(
    frame: pd.DataFrame,
    title: str,
    rolls: list[RollEvent] | None = None,
    years: float = 2.0,
    smas: tuple[int, ...] = (50, 200),
) -> go.Figure:
    """Candles of the quoted front month, the roll-adjusted close, SMAs on the adjusted
    series (so splices don't distort them), roll sessions and volume."""
    start = frame.index[-1] - pd.DateOffset(days=int(365 * years))
    adj = frame["close_adj"] if "close_adj" in frame else frame["close"]
    lines = {n: sma(adj, n) for n in smas}  # full history, then cut: no warm-up gap
    f = frame.loc[start:]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03
    )
    fig.add_trace(
        go.Candlestick(
            x=f.index,
            open=f["open"],
            high=f["high"],
            low=f["low"],
            close=f["close"],
            name="quoted front month",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
        ),
        row=1,
        col=1,
    )
    if "close_adj" in f:
        fig.add_trace(
            go.Scatter(
                x=f.index,
                y=f["close_adj"],
                name="roll-adjusted close",
                line={"width": 1, "dash": "dot", "color": "#6e7781"},
            ),
            row=1,
            col=1,
        )
    for (n, s), color in zip(lines.items(), ("#d29922", "#8250df", "#0969da"), strict=False):
        fig.add_trace(
            go.Scatter(
                x=f.index, y=s.loc[start:], name=f"SMA {n}", line={"width": 1.3, "color": color}
            ),
            row=1,
            col=1,
        )
    switch = [
        pd.Timestamp(r.switch_session)
        for r in rolls or []
        if r.switch_session and pd.Timestamp(r.switch_session) >= start
    ]
    if switch:
        y = f["high"].reindex(switch)
        fig.add_trace(
            go.Scatter(
                x=switch,
                y=y * 1.02,
                mode="markers",
                marker={"symbol": "triangle-down", "size": 7, "color": "#6e7781"},
                name="contract roll",
                hovertemplate="roll %{x|%Y-%m-%d}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "mask_reason" in f:
        bad = f.index[f["mask_reason"] == "nonpositive_price"]
        if len(bad):
            fig.add_trace(
                go.Scatter(
                    x=bad,
                    y=f.loc[bad, "close"],
                    mode="markers",
                    marker={"symbol": "x", "size": 10, "color": DOWN},
                    name="non-positive price",
                ),
                row=1,
                col=1,
            )
    if "volume" in f:
        fig.add_trace(
            go.Bar(
                x=f.index, y=f["volume"], name="volume", marker_color="#afb8c1", showlegend=False
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=560,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": -0.08},
        hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    return fig


# --- forecast -------------------------------------------------------------------------


def skill_caption(h: HorizonForecast) -> str:
    """One line that must accompany any display of this horizon's forecast."""
    s = h.skill
    verdict = "EDGE vs random walk" if s.edge else "no measurable edge"
    return (
        f"{h.horizon}d · {h.model} · skill vs random walk {_pct(s.skill_vs_naive)} · "
        f"band coverage {_pct(s.coverage, signed=False, digits=0)} · "
        f"DM p (Holm) {_p(s.dm_p_adj)} · {verdict}"
    )


def horizon_dates(as_of: date, horizons: list[int]) -> list[pd.Timestamp]:
    """Session dates h business days after `as_of` (weekends skipped, holidays ignored)."""
    return [pd.Timestamp(as_of) + pd.offsets.BDay(h) for h in horizons]


def fan_chart(
    history: pd.Series,
    horizons: list[HorizonForecast],
    as_of: date,
    title: str,
    lookback: int = 120,
) -> go.Figure:
    """Recent closes plus the forecast bands at each horizon, from the last close."""
    if not horizons:
        raise ValueError("no forecast horizons")
    hist = history.dropna().loc[: pd.Timestamp(as_of)].iloc[-lookback:]
    last = float(hist.iloc[-1])
    hs = sorted(horizons, key=lambda h: h.horizon)
    xs = [pd.Timestamp(as_of), *horizon_dates(as_of, [h.horizon for h in hs])]
    qs = sorted(b.quantile for b in hs[0].bands)

    def level(h: HorizonForecast, q: float) -> float:
        b = next(b for b in h.bands if b.quantile == q)
        return float(b.price) if b.price is not None else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="close", line={"color": "#24292f"}))
    pairs = [(qs[i], qs[-1 - i]) for i in range(len(qs) // 2)]
    for depth, (lo, hi) in enumerate(pairs):
        upper = [last, *[level(h, hi) for h in hs]]
        lower = [last, *[level(h, lo) for h in hs]]
        alpha = 0.18 + 0.14 * depth
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=BAND.format(a=alpha),
                line={"width": 0},
                name=f"P{lo * 100:g} to P{hi * 100:g}",
                hoverinfo="skip",
            )
        )
    if 0.5 in qs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[last, *[level(h, 0.5) for h in hs]],
                name="median (P50)",
                mode="lines+markers",
                line={"color": BAND.format(a=1), "dash": "dash"},
                customdata=[["", "", ""]]
                + [[_pct(h.p_up, signed=False, digits=0), h.model, h.skill.verdict] for h in hs],
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<br>P(up) %{customdata[0]}"
                "<br>%{customdata[1]}: %{customdata[2]}<extra></extra>",
            )
        )
    fig.update_layout(
        title={"text": title},
        height=440 + 18 * len(hs),
        margin={"l": 10, "r": 10, "t": 50, "b": 40 + 18 * len(hs)},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom"},
        hovermode="x",
    )
    for i, h in enumerate(hs):  # the skill line is part of the figure, not optional
        fig.add_annotation(
            text=skill_caption(h),
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yanchor="top",
            yshift=-28 - 18 * i,  # below the date axis, one line per horizon
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": "#57606a"},
        )
    return fig


def leaderboard_figure(rows: pd.DataFrame, horizon: int) -> go.Figure:
    """Pinball skill vs the random walk per model (green = significant edge)."""
    r = rows[rows["horizon"] == horizon].sort_values("skill", na_position="first")
    colors = [
        UP if e else (DOWN if (s or 0) < 0 else BENCH)
        for s, e in zip(r["skill"], r["edge"], strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=r["skill"],
            y=r["model"],
            orientation="h",
            marker_color=colors,
            customdata=np.stack([r["coverage"], r["dm_p_adj"], r["n"]], axis=-1)
            if len(r)
            else None,
            hovertemplate="%{y}: skill %{x:.2%}<br>coverage %{customdata[0]:.0%}"
            "<br>DM p (Holm) %{customdata[1]:.3f}<br>origins %{customdata[2]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#6e7781")
    fig.update_layout(
        title=f"{horizon}-day pinball skill vs random walk (out of sample)",
        xaxis_tickformat=".1%",
        height=90 + 32 * max(len(r), 1),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


# --- backtest -------------------------------------------------------------------------


def equity_figure(curves: pd.DataFrame, strategy: str) -> go.Figure:
    """Equity (log scale) vs buy-and-hold, drawdowns, and the held position."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.2], vertical_spacing=0.04
    )
    for col, name, color in (
        ("strategy_equity", strategy, STRATEGY),
        ("benchmark_equity", "buy & hold", BENCH),
    ):
        eq = curves[col]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line={"color": color}), row=1, col=1)
        dd = eq / eq.cummax() - 1
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                name=f"{name} drawdown",
                line={"color": color, "width": 1},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=curves.index,
            y=curves["position"],
            name="position",
            line={"shape": "hv", "color": STRATEGY, "width": 1},
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    both = curves[["strategy_equity", "benchmark_equity"]].to_numpy(float)
    lo, hi = np.nanmin(both), np.nanmax(both)
    ticks = [v for v in (0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50) if lo * 0.8 <= v <= hi * 1.25]
    fig.update_yaxes(
        type="log",
        title_text="growth of 1",
        tickvals=ticks,
        ticktext=[f"{v:g}" for v in ticks],
        row=1,
        col=1,
    )
    fig.update_yaxes(tickformat=".0%", title_text="drawdown", row=2, col=1)
    fig.update_yaxes(title_text="position", range=[-1.05, 1.05], row=3, col=1)
    fig.update_layout(
        height=620,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.04},
    )
    return fig


def cost_by_agent_figure(costs: dict[str, float]) -> go.Figure:
    items = sorted(costs.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=STRATEGY
        )
    )
    fig.update_layout(
        title="LLM cost by agent (USD)",
        xaxis_tickprefix="$",
        height=80 + 28 * max(len(items), 1),
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig
