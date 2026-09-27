"""The final report (Pydantic-validated; saved as JSON, Markdown and HTML).

Every numeric field is filled by code. The LLM-written text fields carry [T#] citations
that resolve to tool results logged in the run's scratchpad.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel

from fa.agents.schemas import AnalystView, ResearcherCase
from fa.forecasting.service import HorizonForecast
from fa.orchestration.decision import Decision

DISCLAIMER = (
    "Research and learning output only, paper trading only. Not investment advice. "
    "Numbers come from logged tool calls; see the scratchpad for their exact sources."
)


class Section(BaseModel):
    title: str
    agent: str | None
    view: AnalystView | None
    text: str  # synthesizer's summary for this section
    key_numbers: dict[str, float | str | None]  # filled by code


class Validation(BaseModel):
    numeric_violations_fixed: int
    sentences_removed: list[str]  # still unsupported after the fix loops
    validator_issues: list[str]
    unresolved_validator_issues: list[str]


class CostSummary(BaseModel):
    usd: float
    tokens: int
    calls: int
    by_agent: dict[str, float]
    budget_usd: float


class Report(BaseModel):
    symbol: str
    name: str
    unit: str
    as_of: date
    generated_at: datetime
    run_id: str
    rating: str
    conviction: float
    decision: Decision
    thesis: str
    forecast: list[HorizonForecast]
    forecast_eval: str  # "walk-forward <start>..<end>, <folds> folds"
    sections: list[Section]
    bull_case: ResearcherCase | None
    bear_case: ResearcherCase | None
    bull_summary: str
    bear_summary: str
    risks: list[str]
    risk_summary: str
    max_exposure: float
    exposure_note: str
    validation: Validation
    data_sources: list[str]
    data_notes: list[str]
    models: dict[str, str]  # agent -> "model (effort), prompt vN"
    cost: CostSummary
    scratchpad: str
    lookahead_warning: str | None = None
    disclaimer: str = DISCLAIMER
