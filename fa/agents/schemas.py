"""Structured outputs of each agent. Text fields carry inline citations like [T3]."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Stance = Literal["bullish", "bearish", "neutral"]


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalystView(_Out):
    stance: Stance
    confidence: float = Field(
        ge=0, le=1, description="How strongly the evidence supports the stance"
    )
    summary: str = Field(description="2-4 sentences; every number cited like [T3]")
    key_points: list[str] = Field(description="3-6 evidence statements, each citing [T#]")
    red_flags: list[str] = Field(description="Evidence against your stance or data problems")
    data_gaps: list[str] = Field(description="Data you needed but that was unavailable")


class ResearcherCase(_Out):
    thesis: str = Field(description="The strongest one-paragraph case for your side")
    arguments: list[str] = Field(description="3-5 arguments, numbers cited like [T3]")
    rebuttals: list[str] = Field(
        description="Answers to the other side's points (empty in round 1)"
    )


class RiskView(_Out):
    risks: list[str] = Field(description="3-6 concrete ways this view could be wrong or lose money")
    suggested_max_exposure: float = Field(
        ge=0, le=1, description="Fraction of equity; may not exceed the code-computed limit"
    )
    exposure_rationale: str


class ValidatorIssue(_Out):
    agent: str
    statement: str = Field(description="The statement quoted exactly")
    problem: str = Field(description="Why the cited data does not support it")


class ValidatorVerdict(_Out):
    issues: list[ValidatorIssue]


class Synthesis(_Out):
    thesis: str = Field(description="One paragraph explaining the rating, citing [T#]")
    fundamentals: str
    technicals: str
    sentiment: str
    macro: str
    forecast: str
    bull_case: str
    bear_case: str
    risks: str
