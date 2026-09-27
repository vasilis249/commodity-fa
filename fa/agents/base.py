"""Agent runner: a manual tool-use loop on the Anthropic Messages API.

- The system prompt comes from prompts/<agent>.md (versioned) plus prompts/_common.md.
- Structured output via `output_config.format` (JSON schema from the Pydantic model);
  if a model rejects it, the runner falls back to "JSON only" instructions and
  validates locally.
- Sonnet 5 / Opus 5.5: adaptive thinking, depth set by `effort`; no temperature, no
  forced tool_choice. The history is append-only (thinking blocks are passed back
  unchanged).
- Every call is budget-checked before and charged after, and logged to the scratchpad.
- `AgentSession.revise(feedback)` continues the same conversation to fix rejected
  output (keeping tool result ids stable and the prompt cache warm).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from fa.config import AppConfig, RoleSpec
from fa.orchestration.budget import CostTracker, Usage
from fa.tools.registry import RunContext, ToolRegistry

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class AgentError(RuntimeError):
    pass


class LLM(Protocol):
    def create(self, agent: str, **params: Any) -> Any: ...


class AnthropicLLM:
    """Thin wrapper so tests can swap in a scripted fake."""

    def __init__(self, client: Any | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=3)
        self.client = client

    def create(self, agent: str, **params: Any) -> Any:
        return self.client.messages.create(**params)


@dataclass(frozen=True)
class Prompt:
    agent: str
    version: str
    text: str


def load_prompt(agent: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    def read(name: str) -> tuple[str, str]:
        raw = (prompts_dir / f"{name}.md").read_text(encoding="utf-8")
        m = FRONT_MATTER.match(raw)
        meta = dict(
            line.split(":", 1) for line in (m.group(1).splitlines() if m else []) if ":" in line
        )
        return meta.get("version", "0").strip(), raw[m.end() :] if m else raw

    common_v, common = read("_common")
    v, body = read(agent)
    return Prompt(agent, f"{v}+common{common_v}", f"{common.strip()}\n\n{body.strip()}")


def _json_from_text(text: str) -> str:
    """The JSON object in a response (tolerates ```json fences in fallback mode)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text


@dataclass
class AgentSession:
    agent: str
    role: RoleSpec
    prompt: Prompt
    output_model: type[BaseModel]
    tools: list[str]
    ctx: RunContext
    registry: ToolRegistry
    llm: LLM
    costs: CostTracker
    max_steps: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    structured: bool = True
    output: BaseModel | None = None

    def _params(self) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self.role.effort}
        if self.structured:
            from anthropic import transform_schema

            output_config["format"] = {
                "type": "json_schema",
                "schema": transform_schema(self.output_model),
            }
        params: dict[str, Any] = {
            "model": self.role.model,
            "max_tokens": self.role.max_tokens,
            "system": [{"type": "text", "text": self.prompt.text}],
            "messages": self.messages,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
        }
        if self.tools:
            params["tools"] = self.registry.schemas(self.tools)
        return params

    def _call(self) -> Any:
        params = self._params()
        est_in = len(json.dumps(params, default=str)) // 3
        reserved = self.costs.preflight(self.role.model, est_in, self.role.max_tokens)
        try:
            response = self.llm.create(self.agent, **params)
        except Exception as exc:
            self.costs.release(reserved)
            if self.structured and _format_unsupported(exc):
                self.structured = False  # fall back to JSON-by-instruction
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Reply with only a JSON object matching this schema:\n"
                        + json.dumps(self.output_model.model_json_schema()),
                    }
                )
                return self._call()
            raise
        usage = Usage.from_response(getattr(response, "usage", None))
        cost = self.costs.charge(self.agent, self.role.model, usage, reserved=reserved)
        self.ctx.scratchpad.log(
            "llm_call",
            agent=self.agent,
            model=self.role.model,
            effort=self.role.effort,
            prompt_version=self.prompt.version,
            stop_reason=getattr(response, "stop_reason", None),
            usage=usage.__dict__,
            cost_usd=round(cost, 6),
            request_id=getattr(response, "_request_id", None),
        )
        return response

    def run(self, user_content: str) -> BaseModel:
        self.messages.append({"role": "user", "content": user_content})
        return self._loop()

    def revise(self, feedback: str) -> BaseModel:
        """Continue the conversation with feedback on the last output."""
        self.messages.append({"role": "user", "content": feedback})
        return self._loop()

    def _loop(self) -> BaseModel:
        for _ in range(self.max_steps):
            response = self._call()
            stop = getattr(response, "stop_reason", None)
            content = list(response.content)
            self.messages.append({"role": "assistant", "content": content})
            if stop == "refusal":
                raise AgentError(f"{self.agent}: the model declined the request")
            if stop == "max_tokens":
                raise AgentError(f"{self.agent}: hit max_tokens ({self.role.max_tokens})")
            if stop == "tool_use":
                results = []
                for block in content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    if block.name not in self.tools:
                        payload, is_error = (
                            {"error": f"tool {block.name} is not available to you"},
                            True,
                        )
                    else:
                        res = self.registry.call(
                            self.ctx, self.agent, block.name, dict(block.input)
                        )
                        payload, is_error = res.payload, res.is_error
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(payload, ensure_ascii=False),
                            "is_error": is_error,
                        }
                    )
                self.messages.append({"role": "user", "content": results})
                continue
            text = next((b.text for b in content if getattr(b, "type", None) == "text"), "")
            try:
                self.output = self.output_model.model_validate_json(_json_from_text(text))
            except ValidationError as exc:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Your answer did not match the required JSON schema: "
                        f"{exc.errors(include_url=False)}. Reply again with only the JSON object.",
                    }
                )
                continue
            self.ctx.scratchpad.log(
                "agent_output", agent=self.agent, output=self.output.model_dump(mode="json")
            )
            return self.output
        raise AgentError(f"{self.agent}: no valid answer within {self.max_steps} steps")


def _format_unsupported(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status == 400 and ("output_config" in text or "format" in text)


def new_session(
    agent: str,
    output_model: type[BaseModel],
    cfg: AppConfig,
    ctx: RunContext,
    registry: ToolRegistry,
    llm: LLM,
    costs: CostTracker,
) -> AgentSession:
    role = cfg.models.roles[cfg.models.agent_roles[agent]]
    spec = cfg.agents.agents[agent]
    return AgentSession(
        agent=agent,
        role=role,
        prompt=load_prompt(agent),
        output_model=output_model,
        tools=list(spec.tools),
        ctx=ctx,
        registry=registry,
        llm=llm,
        costs=costs,
        max_steps=cfg.agents.agent_max_steps,
    )
