"""Tool registry: typed wrappers that expose data/analytics/forecasting to agents.

Each tool has a Pydantic input model (validated before the handler runs) and returns a
JSON-able payload. Every call gets a result id (`T1`, `T2`, ...) that agents must cite
next to every number they state; the scratchpad logs the exact payload the model saw,
so the validator can check each number against it.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from fa.config import AppConfig
from fa.data.http import DataUnavailable
from fa.data.service import DataService
from fa.orchestration.scratchpad import Scratchpad

SIG_DIGITS = 6


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolError(Exception):
    """Raised by handlers for expected failures (data unavailable, not applicable)."""


@dataclass
class RunContext:
    """Everything tools may touch for one run. All data is as of `as_of`."""

    cfg: AppConfig
    svc: DataService
    symbol: str
    as_of: date | None
    scratchpad: Scratchpad
    cache: dict[str, Any] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    def memo(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._lock:  # compute once even when agents call tools in parallel
            if key not in self.cache:
                self.cache[key] = compute()
            return self.cache[key]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[RunContext, Any], Any]

    def schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "input_schema": schema}


def compact(value: Any) -> Any:
    """JSON-able copy with floats rounded to SIG_DIGITS significant digits (and NaN -> None)."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{SIG_DIGITS}g}")
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [compact(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


@dataclass
class ToolResult:
    result_id: str
    payload: Any
    is_error: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, names: list[str]) -> list[dict[str, Any]]:
        unknown = set(names) - set(self._tools)
        if unknown:
            raise KeyError(f"unknown tools: {sorted(unknown)}")
        return [self._tools[n].schema() for n in sorted(names)]  # stable order: cache-friendly

    def call(self, ctx: RunContext, agent: str, name: str, raw_input: dict[str, Any]) -> ToolResult:
        rid = ctx.scratchpad.next_result_id()
        tool = self._tools.get(name)
        error = True
        if tool is None:
            payload: Any = {"error": f"unknown tool {name!r}"}
        else:
            try:
                args = tool.input_model.model_validate(raw_input or {})
                payload = compact(tool.handler(ctx, args))
                error = False
            except ValidationError as exc:
                # no input echo: an agent must not be able to mint citable numbers
                errs = exc.errors(include_url=False, include_input=False, include_context=False)
                payload = {"error": "invalid arguments", "details": [e["msg"] for e in errs]}
            except (ToolError, DataUnavailable) as exc:
                payload = {"error": str(exc)}
        if isinstance(payload, dict):
            payload = {"result_id": rid, **payload}
        else:
            payload = {"result_id": rid, "data": payload}
        ctx.scratchpad.tool_call(agent, name, raw_input or {}, rid, payload, error)
        return ToolResult(rid, payload, error)
