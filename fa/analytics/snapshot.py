"""Numeric snapshot of one instrument at one decision time (no LLM).

Assembles every analytics block into a single Pydantic object. That object is what
`fa analyze --no-llm` prints and what agent tools will expose in Phase 4, so every
number an agent cites traces back to a function here.

Point in time: with `as_of`, prices stop at that session (roll detection included),
and EIA/COT rows count only if `available_at` <= that session's settlement time.
The live futures curve is only available for (roughly) today.
"""

from __future__ import annotations

import calendar
import math
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from pydantic import BaseModel

from fa.analytics import indicators as ind
from fa.analytics import risk, technicals
from fa.analytics.curve import CurveMetrics, curve_metrics
from fa.analytics.models import FiniteModel
from fa.analytics.seasonality import MonthStats, month_stats, monthly_returns
from fa.analytics.supply_demand import (
    cot_positioning,
    crack_spread_321,
    inventory_vs_seasonal,
    percentile_last,
)
from fa.data.http import DataUnavailable
from fa.data.pit import decision_times
from fa.data.service import DataService, FetchInfo

DISCLAIMER = "Research output, paper trading only. Not investment advice."
CRACK_LEGS = ("CL=F", "RB=F", "HO=F")
CURVE_MAX_AGE_DAYS = 5


class _Block(FiniteModel):
    pass


class PriceBlock(_Block):
    """Changes use the roll-adjusted series; masked days count as 0 in multi-day changes."""

    last_date: date
    last_close: float  # actual front-month settle (unadjusted)
    change_1d: float | None
    change_5d: float | None
    change_20d: float | None
    change_60d: float | None
    change_ytd: float | None
    high_52w: float | None  # adjusted series, in today's contract terms
    low_52w: float | None
    pct_from_52w_high: float | None


class LevelOut(_Block):
    price: float
    touches: int
    last_seen: date


class TechBlock(_Block):
    sma20: float | None
    sma50: float | None
    sma200: float | None
    ema20: float | None
    rsi14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_pct_b: float | None
    atr14: float | None
    atr_pct: float | None
    trend: str
    vol_regime: str
    vol_percentile_3y: float | None
    support: list[LevelOut]
    resistance: list[LevelOut]


class RiskBlock(_Block):
    vol_20d: float | None
    vol_1y: float | None
    return_1y: float | None
    sharpe_1y: float | None
    sortino_1y: float | None
    max_drawdown_1y: float | None
    drawdown_from_1y_high: float | None
    var95_1d: float | None
    cvar95_1d: float | None
    position: risk.PositionSize


class SeasonBlock(_Block):
    month_name: str
    stats: MonthStats


class InventoryBlock(_Block):
    name: str
    series_id: str
    period: date
    released_at: datetime
    units: str | None
    value: float
    wow: float | None
    wow_vs_seasonal: float | None
    avg5: float | None
    dev: float | None
    dev_pct: float | None
    min5: float | None
    max5: float | None
    band_pos: float | None


class CotBlock(_Block):
    report_date: date
    released_at: datetime
    mm_net: float
    mm_net_pct_oi: float | None
    mm_net_change_1w: float | None
    mm_net_pctile_3y: float | None
    commercial_net: float | None


class CrackBlock(_Block):
    date: date
    value: float  # USD/bbl
    pctile_1y: float | None
    roll_affected: bool  # a leg is inside a roll window: legs may be different months


class ContractQuote(_Block):
    contract: str
    expiry: date
    settle: float
    settle_date: date


class CurveBlock(_Block):
    contracts: list[ContractQuote]
    metrics: CurveMetrics


class SourceRef(BaseModel):
    provider: str
    key: str
    fetched_at: datetime
    source: str


class Snapshot(BaseModel):
    symbol: str
    name: str
    unit: str
    as_of: date
    decision_time: datetime  # settlement time of `as_of` (UTC)
    generated_at: datetime
    price: PriceBlock
    technicals: TechBlock
    risk: RiskBlock
    seasonality: SeasonBlock | None
    inventories: list[InventoryBlock]
    cot: CotBlock | None
    crack: CrackBlock | None
    curve: CurveBlock | None
    notes: list[str]
    sources: list[SourceRef]
    disclaimer: str = DISCLAIMER


def _last(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.iloc[-1]) if len(x) else float("nan")


def _ref(info: FetchInfo) -> SourceRef:
    return SourceRef(
        provider=info.provider, key=info.key, fetched_at=info.fetched_at, source=info.source
    )


def build_snapshot(
    svc: DataService, symbol: str, as_of: date | None = None, include_curve: bool = True
) -> Snapshot:
    notes: list[str] = []
    sources: list[SourceRef] = []
    pdata = svc.prices(symbol, as_of=as_of)
    inst, frame = pdata.instrument, pdata.frame
    sources.append(_ref(pdata.info))
    session = frame.index[-1]
    decision = decision_times(pd.DatetimeIndex([session]), inst.settle_time, inst.timezone)[0]

    for issue in pdata.quality.issues:
        if issue.severity != "info":
            notes.append(f"data quality [{issue.severity}] {issue.check}: {issue.detail}")
    open_rolls = [e for e in pdata.rolls if e.provisional]
    if open_rolls:
        notes.append(
            f"roll window open (expiry {open_rolls[-1].expiry}): recent returns "
            f"{', '.join(str(d) for d in open_rolls[-1].masked)} are masked and not final"
        )

    close, adj, ret = frame["close"], frame["close_adj"], frame["ret"]
    hi, lo = frame["high_adj"], frame["low_adj"]

    year_start = adj[adj.index < pd.Timestamp(session.year, 1, 1)]
    last_252 = adj.iloc[-252:]
    price = PriceBlock(
        last_date=session.date(),
        last_close=float(close.iloc[-1]),
        # a masked last return (roll, bad price) is unknown, not zero
        change_1d=_last(ind.pct_change_over(adj, 1)) if pd.notna(ret.iloc[-1]) else None,
        change_5d=_last(ind.pct_change_over(adj, 5)),
        change_20d=_last(ind.pct_change_over(adj, 20)),
        change_60d=_last(ind.pct_change_over(adj, 60)),
        change_ytd=float(adj.iloc[-1] / year_start.iloc[-1] - 1) if len(year_start) else None,
        high_52w=float(hi.iloc[-252:].max()),
        low_52w=float(lo.iloc[-252:].min()),
        pct_from_52w_high=float(adj.iloc[-1] / last_252.max() - 1),
    )

    macd = ind.macd(adj)
    bb = ind.bollinger(adj)
    atr = ind.atr(hi, lo, adj, 14)
    trend = technicals.trend_regime(adj)
    vreg, vpct = technicals.vol_regime(ret)
    levels = technicals.support_resistance(hi, lo, adj, _last(atr))
    tech = TechBlock(
        sma20=_last(ind.sma(adj, 20)),
        sma50=_last(ind.sma(adj, 50)),
        sma200=_last(ind.sma(adj, 200)),
        ema20=_last(ind.ema(adj, 20)),
        rsi14=_last(ind.rsi(adj, 14)),
        macd=_last(macd["macd"]),
        macd_signal=_last(macd["macd_signal"]),
        macd_hist=_last(macd["macd_hist"]),
        bb_upper=_last(bb["bb_upper"]),
        bb_lower=_last(bb["bb_lower"]),
        bb_pct_b=_last(bb["bb_pct_b"]),
        atr14=_last(atr),
        atr_pct=_last(atr) / float(adj.iloc[-1]),
        trend=str(trend.iloc[-1]),
        vol_regime=str(vreg.iloc[-1]),
        vol_percentile_3y=_last(vpct),
        support=[
            LevelOut(price=lv.price, touches=lv.touches, last_seen=lv.last_seen.date())
            for lv in levels
            if lv.kind == "support"
        ],
        resistance=[
            LevelOut(price=lv.price, touches=lv.touches, last_seen=lv.last_seen.date())
            for lv in levels
            if lv.kind == "resistance"
        ],
    )

    r1y = risk.to_simple(ret.iloc[-252:])
    vol20 = _last(ind.realized_vol(ret, 20))
    vol1y = risk.annualized_vol(r1y)
    sizing_vol = (
        max(v for v in (vol20, vol1y) if math.isfinite(v))
        if any(math.isfinite(v) for v in (vol20, vol1y))
        else float("nan")
    )
    position = risk.position_size(sizing_vol, int(ret.notna().sum()), svc.cfg.risk)
    position.reasons.insert(0, "sized on the higher of 20-day and 1-year realized vol")
    risk_block = RiskBlock(
        vol_20d=vol20,
        vol_1y=vol1y,
        return_1y=risk.annualized_return(r1y) if len(r1y.dropna()) >= 200 else None,
        sharpe_1y=risk.sharpe(r1y),
        sortino_1y=risk.sortino(r1y),
        max_drawdown_1y=risk.max_drawdown(r1y),
        drawdown_from_1y_high=float(adj.iloc[-1] / last_252.max() - 1),
        var95_1d=risk.var_historical(r1y),
        cvar95_1d=risk.cvar_historical(r1y),
        position=position,
    )

    stats = month_stats(monthly_returns(ret), session.month, session)
    season = SeasonBlock(month_name=calendar.month_name[session.month], stats=stats)

    inventories = _inventories(svc, inst.eia_series, decision, notes, sources)
    cot = _cot(svc, symbol, inst.cot_market_code, decision, notes, sources)
    crack = _crack(svc, symbol, session.date(), notes, sources)
    curve = None
    if include_curve:
        curve = _curve(svc, symbol, session.date(), inst.contract_root, notes, sources)

    return Snapshot(
        symbol=inst.symbol,
        name=inst.name,
        unit=inst.unit,
        as_of=session.date(),
        decision_time=decision.to_pydatetime(),
        generated_at=datetime.now(UTC),
        price=price,
        technicals=tech,
        risk=risk_block,
        seasonality=season,
        inventories=inventories,
        cot=cot,
        crack=crack,
        curve=curve,
        notes=notes,
        sources=sources,
    )


def _inventories(
    svc: DataService,
    series: dict[str, str],
    decision: pd.Timestamp,
    notes: list[str],
    sources: list[SourceRef],
) -> list[InventoryBlock]:
    out = []
    for name, sid in series.items():
        try:
            frame, info = svc.eia(sid, start=date(decision.year - 8, 1, 1))
        except DataUnavailable as exc:
            notes.append(f"EIA {name} unavailable: {exc}")
            continue
        sources.append(_ref(info))
        known = frame[frame["available_at"] <= decision]
        if known.empty:
            notes.append(f"EIA {name}: nothing released by the decision time")
            continue
        inv = inventory_vs_seasonal(known["value"])
        last = inv.iloc[-1]
        out.append(
            InventoryBlock(
                name=name,
                series_id=sid,
                period=inv.index[-1].date(),
                released_at=known["available_at"].iloc[-1].to_pydatetime(),
                units=str(known["units"].iloc[-1]) if "units" in known else None,
                value=float(last["value"]),
                wow=last["wow"],
                wow_vs_seasonal=last["wow_vs_seasonal"],
                avg5=last["avg5"],
                dev=last["dev"],
                dev_pct=last["dev_pct"],
                min5=last["min5"],
                max5=last["max5"],
                band_pos=last["band_pos"],
            )
        )
    return out


def _cot(
    svc: DataService,
    symbol: str,
    code: str | None,
    decision: pd.Timestamp,
    notes: list[str],
    sources: list[SourceRef],
) -> CotBlock | None:
    if not code:
        return None
    try:
        frame, info = svc.cot(symbol, start=date(decision.year - 5, 1, 1))
    except DataUnavailable as exc:
        notes.append(f"COT unavailable: {exc}")
        return None
    sources.append(_ref(info))
    known = frame[frame["available_at"] <= decision]
    if known.empty:
        return None
    pos = cot_positioning(known)
    last = pos.iloc[-1]
    return CotBlock(
        report_date=pos.index[-1].date(),
        released_at=known["available_at"].iloc[-1].to_pydatetime(),
        mm_net=float(last["mm_net"]),
        mm_net_pct_oi=last["mm_net_pct_oi"],
        mm_net_change_1w=last["mm_net_change_1w"],
        mm_net_pctile_3y=last["mm_net_pctile"],
        commercial_net=last["commercial_net"],
    )


def _crack(
    svc: DataService, symbol: str, as_of: date, notes: list[str], sources: list[SourceRef]
) -> CrackBlock | None:
    if symbol not in CRACK_LEGS:
        return None
    legs = {}
    roll_affected = False
    for leg in CRACK_LEGS:
        try:
            data = svc.prices(leg, years=2, as_of=as_of)
        except DataUnavailable as exc:
            notes.append(f"crack spread unavailable ({leg}: {exc})")
            return None
        if leg != symbol:
            sources.append(_ref(data.info))
        legs[leg] = data.frame["close"]
        recent = data.frame["mask_reason"].iloc[-2:]
        roll_affected |= bool((recent == "roll").any())
    crack = crack_spread_321(legs["CL=F"], legs["RB=F"], legs["HO=F"])
    if crack.empty:
        return None
    if roll_affected:
        notes.append("3-2-1 crack: a leg is rolling, so legs may reference different months")
    return CrackBlock(
        date=crack.index[-1].date(),
        value=float(crack.iloc[-1]),
        pctile_1y=percentile_last(crack, 252),
        roll_affected=roll_affected,
    )


def _curve(
    svc: DataService,
    symbol: str,
    as_of: date,
    root: str | None,
    notes: list[str],
    sources: list[SourceRef],
) -> CurveBlock | None:
    if not root:
        return None
    if as_of < svc.now().date() - timedelta(days=CURVE_MAX_AGE_DAYS):
        notes.append("futures curve omitted: only the live curve is available (no history)")
        return None
    try:
        curve, infos = svc.curve(symbol, as_of=as_of)
    except DataUnavailable as exc:
        notes.append(f"futures curve unavailable: {exc}")
        return None
    sources.extend(_ref(i) for i in infos)
    metrics = curve_metrics(curve)
    if metrics is None:
        return None
    return CurveBlock(
        contracts=[
            ContractQuote(
                contract=r.contract, expiry=r.expiry, settle=r.settle, settle_date=r.settle_date
            )
            for r in curve.itertuples()
        ],
        metrics=metrics,
    )
