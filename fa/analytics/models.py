"""Shared base for analytics outputs: non-finite floats become None.

NaN/inf are not valid JSON and never compare equal, so every analytics model stores
"not computable" as None instead.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator


class FiniteModel(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _nan_to_none(cls, v: object) -> object:
        return None if isinstance(v, float) and not math.isfinite(v) else v
