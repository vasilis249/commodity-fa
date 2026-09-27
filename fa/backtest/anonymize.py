"""Anonymized tools for LLM-agent backtests (design rule 6: fight the LLM's memory).

The model must not be able to tell *which* asset or *when*. Every payload is built from
an allow-list of fields, never by deleting fields from a full payload:
- price levels are rebased to 100 with a single factor fixed at the first decision
  (known at every later decision, so it is causal);
- no dates, names, tickers, units, contract codes, data sources or calendar months;
- quantities with telltale scales (barrels, contracts, USD/bbl) are shown only as
  ratios, percentiles or positions within a band;
- news, macro series, benchmark returns and seasonality are not offered: each can
  reveal the calendar date.
Payloads are produced inside `ToolRegistry.call`, before `compact` and the scratchpad,
so the claim checker validates exactly what the model saw.

Residual risk: the shape of the price path can still hint at famous episodes; reports
from these backtests say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fa.analytics.snapshot import Snapshot
from fa.config import ForecastingConfig
from fa.forecasting.base import qcol
from fa.forecasting.evaluate import EvalResult, diebold_mariano, dm_block, mean_pinball
from fa.tools.registry import NoArgs, RunContext, Tool, ToolError, ToolRegistry

INVENTORY_LABELS = "ABCDEFGH"


@dataclass
class AnonState:
    """Per-decision inputs (set by the backtest before each agent run)."""

    snapshot: Snapshot
    factor: float  # 100 / reference price at the first decision
    forecast_rows: dict[int, dict[str, float]]  # horizon -> q.., p_up at this origin
    past_skill: dict[int, dict[str, float | None]]  # horizon -> skill over settled origins
    sessions_to_roll: int | None
    extra: dict[str, Any] = field(default_factory=dict)


def _state(ctx: RunContext) -> AnonState:
    st = ctx.cache.get("anon")
    if not isinstance(st, AnonState):
        raise ToolError("anonymized state not set")
    return st


def _lvl(st: AnonState, x: float | None) -> float | None:
    return None if x is None else x * st.factor


def price_technicals(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    st = _state(ctx)
    p, t = st.snapshot.price, st.snapshot.technicals
    return {
        "price_index": {
            "last": _lvl(st, p.last_close),
            "change_1d": p.change_1d,
            "change_5d": p.change_5d,
            "change_20d": p.change_20d,
            "change_60d": p.change_60d,
            "high_52w": _lvl(st, p.high_52w),
            "low_52w": _lvl(st, p.low_52w),
            "pct_from_52w_high": p.pct_from_52w_high,
        },
        "technicals": {
            "sma20": _lvl(st, t.sma20),
            "sma50": _lvl(st, t.sma50),
            "sma200": _lvl(st, t.sma200),
            "rsi14": t.rsi14,
            "macd_hist_pct_of_price": (t.macd_hist / p.last_close)
            if t.macd_hist is not None and p.last_close
            else None,
            "bb_pct_b": t.bb_pct_b,
            "atr_pct": t.atr_pct,
            "trend": t.trend,
            "vol_regime": t.vol_regime,
            "vol_percentile_3y": t.vol_percentile_3y,
            "support": [_lvl(st, lv.price) for lv in t.support],
            "resistance": [_lvl(st, lv.price) for lv in t.resistance],
        },
        "note": "price index rebased to 100 at the start of the test; names and dates are hidden",
    }


def risk_metrics(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    r = _state(ctx).snapshot.risk
    return {
        "vol_20d": r.vol_20d,
        "vol_1y": r.vol_1y,
        "sharpe_1y": r.sharpe_1y,
        "max_drawdown_1y": r.max_drawdown_1y,
        "drawdown_from_1y_high": r.drawdown_from_1y_high,
        "var95_1d": r.var95_1d,
        "suggested_max_exposure": r.position.fraction,
    }


def inventories(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    invs = _state(ctx).snapshot.inventories
    if not invs:
        raise ToolError("no inventory data for this asset")
    return {
        "series": [
            {
                "name": f"inventory series {INVENTORY_LABELS[i]}",
                "vs_5y_average": inv.dev_pct,
                "position_in_5y_band": inv.band_pos,
            }
            for i, inv in enumerate(invs[: len(INVENTORY_LABELS)])
        ],
        "note": "position_in_5y_band: 0 = at the 5-year low, 1 = at the 5-year high",
    }


def positioning(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    cot = _state(ctx).snapshot.cot
    if cot is None:
        raise ToolError("no positioning data for this asset")
    return {
        "speculators_net_share_of_open_interest": cot.mm_net_pct_oi,
        "speculators_net_percentile_3y": cot.mm_net_pctile_3y,
    }


def processing_margin(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    crack = _state(ctx).snapshot.crack
    if crack is None:
        raise ToolError("no processing-margin data for this asset")
    return {"margin_percentile_1y": crack.pctile_1y}


def forecast(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    st = _state(ctx)
    if not st.forecast_rows:
        raise ToolError("no forecast at this date")
    return {
        "horizons": [
            {"horizon_days": h, **row, "skill_so_far": st.past_skill.get(h)}
            for h, row in sorted(st.forecast_rows.items())
        ],
        "note": (
            "q* are log-return quantiles; skill_so_far compares this model with a random "
            "walk only on forecasts whose outcomes were known before this date"
        ),
    }


def roll_calendar(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    st = _state(ctx)
    return {"sessions_to_next_contract_roll": st.sessions_to_roll}


ANON_TOOLS: dict[str, tuple[str, Any]] = {
    "price_technicals": (
        "Rebased price index, recent changes, moving averages, RSI, Bollinger %b, ATR %, trend and volatility regime, support/resistance.",
        price_technicals,
    ),
    "risk_metrics": (
        "Volatility, drawdowns, VaR and the code-computed suggested max exposure.",
        risk_metrics,
    ),
    "inventories": (
        "Weekly inventory statistics versus their prior 5-year seasonal band (as ratios).",
        inventories,
    ),
    "positioning": (
        "Speculators' net position as a share of open interest and its 3-year percentile.",
        positioning,
    ),
    "processing_margin": (
        "1-year percentile of the downstream processing margin.",
        processing_margin,
    ),
    "forecast": (
        "Model forecast quantiles and P(up) with the model's skill on already-settled forecasts.",
        forecast,
    ),
    "roll_calendar": ("Sessions until the next futures contract roll.", roll_calendar),
}

ANON_AGENT_TOOLS = {
    "supply_demand_analyst": ["inventories", "positioning", "processing_margin"],
    "technical_analyst": ["price_technicals", "roll_calendar"],
    "forecast_interpreter": ["forecast"],
    "risk_reviewer": ["risk_metrics", "price_technicals"],
}


def build_anon_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name, (desc, fn) in ANON_TOOLS.items():
        reg.register(Tool(name, desc, NoArgs, fn))
    return reg


def forecast_rows_at(ev: EvalResult, model: str, date: pd.Timestamp) -> dict[int, dict[str, float]]:
    out = {}
    for h, frame in ev.predictions[model].items():
        if date in frame.index:
            row = frame.loc[date]
            out[h] = {k: float(row[k]) for k in frame.columns if np.isfinite(row[k])}
    return out


def past_skill(
    ev: EvalResult, model: str, pos: int, cfg: ForecastingConfig
) -> dict[int, dict[str, float | None]]:
    """Skill vs the naive baseline on origins whose labels were complete by row `pos`."""
    out: dict[int, dict[str, float | None]] = {}
    base = cfg.skill.baseline
    for h, y in ev.realized.items():
        settled = (ev.positions[h] + h <= pos) & np.isfinite(y.to_numpy())
        if settled.sum() < 20 or model not in ev.predictions or base not in ev.predictions:
            out[h] = None  # type: ignore[assignment]
            continue
        yv = y.to_numpy()[settled]
        loss = mean_pinball(yv, ev.predictions[model][h][settled], cfg.quantiles)
        base_loss = mean_pinball(yv, ev.predictions[base][h][settled], cfg.quantiles)
        pin, bpin = float(np.mean(loss)), float(np.mean(base_loss))
        out[h] = {
            "forecasts_scored": int(settled.sum()),
            "pinball_skill_vs_random_walk": 1 - pin / bpin if bpin > 0 else None,
            "dm_p_value": diebold_mariano(loss - base_loss, dm_block(h, cfg)),
        }
    return out


__all__ = [
    "ANON_AGENT_TOOLS",
    "AnonState",
    "build_anon_registry",
    "forecast_rows_at",
    "past_skill",
    "qcol",
]
