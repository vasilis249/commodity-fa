"""A scripted stand-in for the Anthropic client, plus an offline data service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fa.config import AppConfig, Secrets
from fa.data.cache import DiskCache
from fa.data.service import DataService

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


def _usage(inp: int = 1500, out: int = 300) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _resp(stop: str, content: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop, content=content, usage=_usage(), _request_id="req_fake"
    )


def _text(s: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=s)


def _first_number(payload: Any, path: str = "") -> tuple[str, float] | None:
    """A (field path, value) pair from a tool payload, skipping ids and booleans."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k == "result_id":
                continue
            hit = _first_number(v, f"{path}.{k}" if path else k)
            if hit:
                return hit
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            hit = _first_number(v, f"{path}[{i}]")
            if hit:
                return hit
    elif isinstance(payload, float) and not isinstance(payload, bool) and abs(payload) >= 13:
        return path, payload
    return None


def _results_in(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and not b.get("is_error"):
                    out.append(json.loads(b["content"]))
    return out


def _cited_sentences(text: str) -> list[str]:
    """Cited sentences from the JSON embedded in an input message (analyst views etc.)."""
    from fa.agents.claims import CITE, SENTENCE

    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return []
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            found.extend(s.strip() for s in SENTENCE.split(v) if CITE.search(s))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(obj)
    return list(dict.fromkeys(found))


@dataclass
class FakeLLM:
    """Behaves like a careful analyst unless told otherwise.

    lie_once: agents whose first answer contains an unsupported number.
    lie_always: agents that never fix it (the sentence must get stripped).
    refuse: agents whose calls end with stop_reason="refusal".
    flag_once: agents the validator flags once.
    reject_format: raise a 400 on output_config.format the first time (fallback test).
    """

    lie_once: set[str] = field(default_factory=set)
    lie_always: set[str] = field(default_factory=set)
    refuse: set[str] = field(default_factory=set)
    flag_once: set[str] = field(default_factory=set)
    flag_always: set[str] = field(default_factory=set)  # validator never accepts these
    flag_stance: set[str] = field(default_factory=set)  # validator rejects their stance
    reject_format: bool = False
    stance: str = "bullish"
    # research_agent: which tools it calls, and how it answers
    research_tools: tuple[str, ...] = ("price_technicals",)
    research_mode: str = "honest"  # honest | lie_once | lie_always | decline
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    _flagged: set[str] = field(default_factory=set)

    def create(self, agent: str, **params: Any) -> Any:
        self.calls.append((agent, params))
        if self.reject_format and "format" in params.get("output_config", {}):
            self.reject_format = False
            err = RuntimeError("400 output_config.format is not supported for this model")
            err.status_code = 400  # type: ignore[attr-defined]
            raise err
        if agent in self.refuse:
            return _resp("refusal", [])
        messages = params["messages"]
        tools = [t["name"] for t in params.get("tools", [])]
        if agent == "research_agent":
            tools = [t for t in self.research_tools if t in tools]
        if (
            tools
            and not _results_in(messages)
            and not any(
                isinstance(m["content"], list)
                and any(getattr(b, "type", "") == "tool_use" for b in m["content"])
                for m in messages
                if m["role"] == "assistant"
            )
        ):
            blocks = [
                SimpleNamespace(type="tool_use", id=f"tu_{agent}_{n}", name=n, input={})
                for n in tools
            ]
            return _resp("tool_use", blocks)
        last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        feedback = isinstance(last_user, str) and (
            "code check" in last_user or "reviewer" in last_user
        )
        return _resp("end_turn", [_text(json.dumps(self._answer(agent, messages, feedback)))])

    # --- answers -----------------------------------------------------------------------

    def _evidence(self, messages: list[dict[str, Any]]) -> list[str]:
        sentences = []
        for payload in _results_in(messages):
            hit = _first_number(payload)
            if hit:
                sentences.append(f"The {hit[0]} reading is {hit[1]!r} [{payload['result_id']}].")
        if not sentences:  # agents without tools: reuse cited sentences from the inputs
            for m in messages:
                if m["role"] == "user" and isinstance(m["content"], str):
                    sentences += _cited_sentences(m["content"])
        return sentences[:4] or ["The evidence is qualitative."]

    def _lie(self, agent: str, feedback: bool) -> list[str]:
        if agent in self.lie_always or (agent in self.lie_once and not feedback):
            return ["Price is 12345.67 [T1]."]
        return []

    def _research(self, messages: list[dict[str, Any]], feedback: bool) -> dict[str, Any]:
        results = _results_in(messages)
        hit = next((h for p in results if (h := _first_number(p))), None)
        rid = next((p["result_id"] for p in results if _first_number(p)), None)
        if self.research_mode == "decline" or hit is None:
            return {
                "plan": ["check the tools"],
                "answerable": False,
                "value": None,
                "unit": None,
                "choice": None,
                "answer": "The tools cannot answer this.",
                "citations": [],
            }
        lie = self.research_mode == "lie_always" or (
            self.research_mode == "lie_once" and not feedback
        )
        value = 12345.67 if lie else hit[1]
        return {
            "plan": ["read the price tool"],
            "answerable": True,
            "value": value,
            "unit": "USD/bbl",
            "choice": None,
            "answer": f"The {hit[0]} reading is {value!r} [{rid}].",
            "citations": [rid],
        }

    def _answer(self, agent: str, messages: list[dict[str, Any]], feedback: bool) -> dict[str, Any]:
        if agent == "research_agent":
            return self._research(messages, feedback)
        ev = self._evidence(messages) + self._lie(agent, feedback)
        if agent == "validator":
            issues = []
            payload = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
            for st in payload["statements"]:
                is_stance = st["statement"].startswith("stance:")
                if st["agent"] in self.flag_stance and is_stance:
                    issues.append(
                        {
                            "agent": st["agent"],
                            "statement": st["statement"],
                            "problem": "evidence is mixed",
                        }
                    )
                    continue
                if st["agent"] in self.flag_always and not is_stance:
                    issues.append(
                        {
                            "agent": st["agent"],
                            "statement": st["statement"],
                            "problem": "reversed sign",
                        }
                    )
                    continue
                if is_stance:
                    continue
                if st["agent"] in self.flag_once and st["agent"] not in self._flagged:
                    self._flagged.add(st["agent"])
                    issues.append(
                        {
                            "agent": st["agent"],
                            "statement": st["statement"],
                            "problem": "misread field",
                        }
                    )
            return {"issues": issues}
        if agent.endswith("_researcher"):
            return {"thesis": " ".join(ev[:2]), "arguments": ev, "rebuttals": []}
        if agent == "risk_reviewer":
            return {
                "risks": ev,
                "suggested_max_exposure": 0.9,
                "exposure_rationale": "volatility is high",
            }
        if agent == "synthesizer":
            s = " ".join(ev[:2])
            return {
                k: s
                for k in (
                    "thesis",
                    "fundamentals",
                    "technicals",
                    "sentiment",
                    "macro",
                    "forecast",
                    "bull_case",
                    "bear_case",
                    "risks",
                )
            }
        return {
            "stance": self.stance,
            "confidence": 0.6,
            "summary": " ".join(ev[:2]),
            "key_points": ev,
            "red_flags": [],
            "data_gaps": [],
        }


class OfflineNews:
    requests_made = 0

    def fetch_feed(self, feed: str) -> Any:
        import pandas as pd

        from fa.data.providers.news import parse_rss
        from tests.conftest import FIXTURES

        frame = parse_rss((FIXTURES / "rss_google_news.xml").read_text(), feed=feed)
        frame["published_at"] = NOW - pd.to_timedelta(range(len(frame)), unit="h")
        return frame


def offline_service(cfg: AppConfig, tmp: Path) -> DataService:
    from tests.integration.test_snapshot import KeyedProvider, _cot, _prices, _weekly

    inst = cfg.data.instrument("CL=F")
    assert inst is not None
    eia = {sid: _weekly("W-FRI", 400 + 50 * i, i) for i, sid in enumerate(inst.eia_series.values())}
    return DataService(
        cfg,
        secrets=Secrets(_env_file=None),
        cache=DiskCache(tmp / "cache"),
        providers={
            "yahoo": KeyedProvider(_prices()),
            "eia": KeyedProvider(eia),
            "cftc": KeyedProvider({"067651": _cot()}),
        },
        news=OfflineNews(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )
