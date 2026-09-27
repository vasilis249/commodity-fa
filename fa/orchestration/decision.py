"""The rating is computed in code, never by an LLM (design rule 1).

score = sum_i weight_i x stance_i x confidence_i, stance = +1 bullish / -1 bearish / 0.
The forecast component uses the forecast service directly: it counts only at horizons
where a model beats the random walk out of sample (2 x P(up) - 1, averaged over those
horizons); without an edge it contributes 0 and its weight still counts, which pulls
conviction down, as it should.
"""

from __future__ import annotations

from typing import Literal

from fa.agents.schemas import AnalystView
from fa.analytics.models import FiniteModel
from fa.config import AgentsConfig
from fa.forecasting.service import ForecastReport

Rating = Literal["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
STANCE = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


class Component(FiniteModel):
    name: str
    weight: float
    signal: float  # in [-1, 1]
    contribution: float  # weight x signal
    note: str


class Decision(FiniteModel):
    rating: Rating
    score: float  # in [-1, 1]
    conviction: float  # |score| / sum of weights, in [0, 1]
    components: list[Component]
    forecast_edge: bool


def forecast_signal(fc: ForecastReport | None) -> tuple[float, bool, str]:
    if fc is None:
        return 0.0, False, "forecast unavailable"
    edged = [h for h in fc.horizons if h.skill.edge and h.p_up is not None]
    if not edged:
        return 0.0, False, "no measurable edge vs random walk: contributes 0"
    sig = sum(2 * float(h.p_up or 0.5) - 1 for h in edged) / len(edged)
    return max(-1.0, min(1.0, sig)), True, f"edge at {', '.join(f'{h.horizon}d' for h in edged)}"


def rate(score: float, cfg: AgentsConfig) -> Rating:
    t = cfg.rating_thresholds
    if score >= t.strong:
        return "Strong Buy"
    if score >= t.weak:
        return "Buy"
    if score <= -t.strong:
        return "Strong Sell"
    if score <= -t.weak:
        return "Sell"
    return "Hold"


def decide(views: dict[str, AnalystView], fc: ForecastReport | None, cfg: AgentsConfig) -> Decision:
    components = []
    for name, weight in cfg.decision_weights.items():
        if name == "forecast":
            sig, _, note = forecast_signal(fc)
        elif name in views:
            v = views[name]
            sig, note = (
                STANCE[v.stance] * v.confidence,
                f"{v.stance}, confidence {v.confidence:.2f}",
            )
        else:
            sig, note = 0.0, "agent disabled or failed: contributes 0"
        components.append(
            Component(name=name, weight=weight, signal=sig, contribution=weight * sig, note=note)
        )
    total_w = sum(c.weight for c in components) or 1.0
    score = sum(c.contribution for c in components) / total_w
    _, edge, _ = forecast_signal(fc)
    return Decision(
        rating=rate(score, cfg),
        score=score,
        conviction=abs(score),
        components=components,
        forecast_edge=edge,
    )
