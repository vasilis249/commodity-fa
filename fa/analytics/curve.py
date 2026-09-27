"""Futures term structure metrics from a snapshot of listed contracts. Pure."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from fa.analytics.models import FiniteModel

Structure = Literal["backwardation", "contango", "flat", "mixed"]


class CurveMetrics(FiniteModel):
    front: str
    second: str
    m1_m2_spread: float  # front - second, price units
    m1_m2_pct: float | None  # (front - second) / second
    roll_yield_ann: float | None  # annualized log(front/second) over the expiry gap
    front_to_back_ann: float | None  # same, front vs the last listed contract
    structure: Structure
    contracts: int


def curve_metrics(curve: pd.DataFrame, flat_tolerance: float = 0.001) -> CurveMetrics | None:
    """`curve` columns: contract, expiry (date), settle; ordered by expiry.

    Positive roll yield = backwardation (a long position gains from rolling up the
    curve if prices do not move).
    """
    c = curve.dropna(subset=["settle"]).sort_values("expiry")
    c = c[c["settle"] > 0]
    if len(c) < 2:
        return None
    f1, f2, fn = c.iloc[0], c.iloc[1], c.iloc[-1]

    def ann(a: pd.Series, b: pd.Series) -> float:
        years = (pd.Timestamp(b["expiry"]) - pd.Timestamp(a["expiry"])).days / 365.25
        return float(np.log(a["settle"] / b["settle"]) / years) if years > 0 else float("nan")

    diffs = np.diff(c["settle"].to_numpy())
    rel = diffs / c["settle"].to_numpy()[:-1]
    if np.all(np.abs(rel) <= flat_tolerance):
        structure: Structure = "flat"
    elif np.all(rel <= flat_tolerance):
        structure = "backwardation"
    elif np.all(rel >= -flat_tolerance):
        structure = "contango"
    else:
        structure = "mixed"
    return CurveMetrics(
        front=str(f1["contract"]),
        second=str(f2["contract"]),
        m1_m2_spread=float(f1["settle"] - f2["settle"]),
        m1_m2_pct=float((f1["settle"] - f2["settle"]) / f2["settle"]),
        roll_yield_ann=ann(f1, f2),
        front_to_back_ann=ann(f1, fn),
        structure=structure,
        contracts=len(c),
    )
