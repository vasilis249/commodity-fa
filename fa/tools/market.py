"""The tools agents can call. Every number they return comes from tested code:
the numeric snapshot (fa.analytics.snapshot), the forecast service, or the data layer.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fa.analytics.snapshot import Snapshot, build_snapshot
from fa.config import RollRule
from fa.data.calendars import calendar_for_exchange
from fa.data.http import DataUnavailable
from fa.data.pit import release_date
from fa.data.rolls import contract_code, expiries
from fa.forecasting.service import ForecastReport, run_forecast
from fa.tools.registry import NoArgs, RunContext, Tool, ToolError, ToolRegistry

HURRICANE_SEASON = ((6, 1), (11, 30))


def _snapshot(ctx: RunContext) -> Snapshot:
    return ctx.memo("snapshot", lambda: build_snapshot(ctx.svc, ctx.symbol, as_of=ctx.as_of))


def _forecast(ctx: RunContext) -> ForecastReport:
    return ctx.memo("forecast", lambda: run_forecast(ctx.svc, ctx.symbol, as_of=ctx.as_of))


def _require(block: Any, what: str) -> Any:
    if block is None:
        raise ToolError(f"{what} is not available for this instrument or date")
    return block


def price_technicals(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "unit": s.unit, "price": s.price, "technicals": s.technicals}


def risk_metrics(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "risk": s.risk}


def risk_limits(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    return {"limits": ctx.cfg.risk.model_dump(mode="json"), "note": "hard limits enforced in code"}


def seasonality(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "seasonality": _require(s.seasonality, "seasonality")}


def inventories(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    if not s.inventories:
        raise ToolError("no EIA inventory series configured or released for this instrument")
    return {"as_of": s.as_of, "inventories": s.inventories}


def cot_positioning(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "cot": _require(s.cot, "CFTC positioning")}


def crack_spread(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "crack_321_usd_bbl": _require(s.crack, "the 3-2-1 crack spread")}


def term_structure(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {"as_of": s.as_of, "curve": _require(s.curve, "the futures curve")}


def data_quality(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    s = _snapshot(ctx)
    return {
        "as_of": s.as_of,
        "notes": s.notes,
        "sources": sorted({f"{x.provider}:{x.key}" for x in s.sources}),
    }


def forecast(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    f = _forecast(ctx)
    return {
        "as_of": f.as_of,
        "last_close": f.last_close,
        "horizons": [
            {
                "horizon_days": h.horizon,
                "model": h.model,
                "bands": h.bands,
                "p_up": h.p_up,
                "has_edge_vs_random_walk": h.skill.edge,
                "verdict": h.skill.verdict,
                "interval_coverage": h.skill.coverage,
            }
            for h in f.horizons
        ],
        "notes": f.notes,
    }


def forecast_leaderboard(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    f = _forecast(ctx)
    return {
        "eval_period": [f.eval_start, f.eval_end],
        "folds": f.folds,
        "rows": [
            r.model_dump(include={"model", "horizon", "skill", "coverage", "dm_p_adj", "edge"})
            for r in f.leaderboard
        ],
    }


class NewsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=40, description="Most recent headlines to return")


def news_headlines(ctx: RunContext, args: NewsArgs) -> dict[str, Any]:
    if ctx.as_of is not None and ctx.as_of < ctx.svc.now().date() - timedelta(days=3):
        raise ToolError("news is live-only: RSS feeds cannot be replayed for a past date")
    frame, infos = ctx.svc.news(ctx.symbol)
    rows = frame.head(args.limit)
    return {
        "headlines": [
            {"published_at": r.published_at.isoformat(), "title": r.title, "source": r.source}
            for r in rows.itertuples()
        ],
        "feeds": len(infos),
        "note": "headlines only; judge tone yourself and cite the headline text",
    }


def event_calendar(ctx: RunContext, _: NoArgs) -> dict[str, Any]:
    inst = ctx.svc.instrument(ctx.symbol)
    ref = ctx.as_of or ctx.svc.now().date()
    events: list[dict[str, Any]] = []
    rel = ctx.cfg.data.releases
    last_friday = ref - timedelta(days=(ref.weekday() - 4) % 7)
    for key, label in (
        ("eia_petroleum_weekly", "EIA Weekly Petroleum Status Report"),
        ("eia_gas_storage_weekly", "EIA Weekly Natural Gas Storage Report"),
    ):
        d = release_date(last_friday, rel[key])
        events.append(
            {
                "date": d if d >= ref else release_date(last_friday + timedelta(days=7), rel[key]),
                "event": label,
            }
        )
    last_tuesday = ref - timedelta(days=(ref.weekday() - 1) % 7)
    d = release_date(last_tuesday, rel["cftc_cot"])
    events.append(
        {
            "date": d
            if d >= ref
            else release_date(last_tuesday + timedelta(days=7), rel["cftc_cot"]),
            "event": "CFTC Commitments of Traders",
        }
    )
    if inst.roll_rule is not RollRule.NONE:
        cal = calendar_for_exchange(inst.exchange)
        nxt = expiries(inst.roll_rule, ref, ref + timedelta(days=70), cal)
        for ts, delivery in nxt.items():
            y, m = (int(x) for x in str(delivery).split("-"))
            code = contract_code(inst.contract_root or inst.symbol, y, m)
            events.append(
                {"date": ts.date(), "event": f"{code} last trading day (front-month roll)"}
            )
    (sm, sd), (em, ed) = HURRICANE_SEASON
    in_season = date(ref.year, sm, sd) <= ref <= date(ref.year, em, ed)
    return {
        "as_of": ref,
        "events": sorted(events, key=lambda e: e["date"]),
        "atlantic_hurricane_season": in_season,
        "not_covered": ["OPEC+ meetings", "FOMC meetings"],
    }


def macro_series(ctx: RunContext, args: NoArgs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sid in ctx.cfg.data.fred_macro.items():
        try:
            frame, _ = ctx.svc.fred(sid)
        except DataUnavailable as exc:
            raise ToolError(f"FRED unavailable ({exc}); use relative_strength instead") from exc
        cutoff = pd.Timestamp(ctx.as_of) if ctx.as_of else None
        known = (
            frame if cutoff is None else frame[frame["available_at"] <= cutoff.tz_localize("UTC")]
        )
        v = known["value"].dropna()
        if v.empty:
            continue
        out[name] = {
            "series": sid,
            "last_date": v.index[-1].date(),
            "last": float(v.iloc[-1]),
            "change_20_obs": float(v.iloc[-1] - v.iloc[-21]) if len(v) > 20 else None,
        }
    return {"macro": out}


class RelStrengthArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmarks: list[str] = Field(
        default=["SPY", "XLE", "DX-Y.NYB"],
        max_length=4,
        description="Yahoo tickers to compare against (DX-Y.NYB = US dollar index)",
    )


def relative_strength(ctx: RunContext, args: RelStrengthArgs) -> dict[str, Any]:
    def perf(sym: str) -> dict[str, float | None]:
        adj = ctx.svc.prices(sym, years=2, as_of=ctx.as_of).frame["close_adj"]
        return {
            f"return_{n}d": float(adj.iloc[-1] / adj.iloc[-1 - n] - 1) if len(adj) > n else None
            for n in (20, 60, 120)
        }

    own = perf(ctx.symbol)
    rows = {ctx.symbol: own}
    for b in args.benchmarks:
        try:
            rows[b] = perf(b)
        except DataUnavailable as exc:
            rows[b] = {"error": str(exc)}  # type: ignore[dict-item]
    return {"returns": rows, "note": "simple returns on roll-adjusted closes"}


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    specs: list[tuple[str, str, type[BaseModel], Callable[[RunContext, Any], Any]]] = [
        (
            "price_technicals",
            "Latest price, 1d-YTD changes, 52-week range, SMA/EMA, RSI, MACD, Bollinger, ATR, trend and volatility regime, support/resistance.",
            NoArgs,
            price_technicals,
        ),
        (
            "risk_metrics",
            "1-year volatility, Sharpe, Sortino, max drawdown, VaR/CVaR and the code-computed suggested max exposure with its reasons.",
            NoArgs,
            risk_metrics,
        ),
        (
            "risk_limits",
            "Hard risk limits from config (max exposure, vol target, drawdown stop).",
            NoArgs,
            risk_limits,
        ),
        (
            "seasonality",
            "This calendar month's historical return statistics (past years only).",
            NoArgs,
            seasonality,
        ),
        (
            "inventories",
            "EIA weekly inventories/production/utilization vs the prior 5-year seasonal band, as released by the decision time.",
            NoArgs,
            inventories,
        ),
        (
            "cot_positioning",
            "CFTC managed-money net position, share of open interest, weekly change and 3-year percentile (released data only).",
            NoArgs,
            cot_positioning,
        ),
        (
            "crack_spread",
            "3-2-1 refining crack spread in USD/bbl with its 1-year percentile (crude and products only).",
            NoArgs,
            crack_spread,
        ),
        (
            "term_structure",
            "Futures curve: contract settles, M1-M2 spread, annualized roll yield, contango/backwardation.",
            NoArgs,
            term_structure,
        ),
        (
            "data_quality",
            "Data-quality warnings and data sources behind the numbers.",
            NoArgs,
            data_quality,
        ),
        (
            "forecast",
            "Probabilistic forecast (P10/P50/P90, P(up)) for 1/5/20 days and whether any model beats a random walk out of sample.",
            NoArgs,
            forecast,
        ),
        (
            "forecast_leaderboard",
            "Out-of-sample leaderboard of all forecasting models vs the naive random walk.",
            NoArgs,
            forecast_leaderboard,
        ),
        (
            "news_headlines",
            "Recent headlines (live only) from Google News and energy feeds.",
            NewsArgs,
            news_headlines,
        ),
        (
            "event_calendar",
            "Upcoming scheduled events: EIA reports, CFTC release, contract expiries, hurricane season.",
            NoArgs,
            event_calendar,
        ),
        (
            "macro_series",
            "FRED macro series (dollar index, rates, breakevens, industrial production), first-release values.",
            NoArgs,
            macro_series,
        ),
        (
            "relative_strength",
            "20/60/120-day returns of this instrument vs benchmarks (default SPY, XLE, dollar index).",
            RelStrengthArgs,
            relative_strength,
        ),
    ]
    for name, desc, model, fn in specs:
        reg.register(Tool(name, desc, model, fn))
    return reg
