"""JSONL scratchpad: one line per event, so every claim can be traced to its data.

File: `.runs/<UTC timestamp>_<run id>.jsonl`. Events: run_start, tool_call (full result,
exactly as the model saw it), llm_call (model, effort, usage, cost), agent_output,
check (numeric/validator findings), decision, error, run_end.
Secrets never enter: every line passes through `redact`.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fa.data.http import redact


def _default(o: object) -> object:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


class Scratchpad:
    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = runs_dir / f"{stamp}_{self.run_id}.jsonl"
        self._seq = 0
        self._lock = threading.RLock()  # agents may run in parallel threads
        self.results: dict[str, Any] = {}  # result id -> the exact payload the model saw

    def log(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._seq += 1
            line = {"seq": self._seq, "ts": datetime.now(UTC).isoformat(), "event": event, **fields}
            text = redact(json.dumps(line, default=_default, ensure_ascii=False))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

    def next_result_id(self) -> str:
        """Reserve a result id (thread-safe; ids are never reused)."""
        with self._lock:
            rid = f"T{len(self.results) + 1}"
            self.results[rid] = None
            return rid

    def tool_call(
        self, agent: str, tool: str, args: dict[str, Any], result_id: str, payload: Any, error: bool
    ) -> None:
        self.results[result_id] = payload
        self.log(
            "tool_call",
            agent=agent,
            tool=tool,
            args=args,
            result_id=result_id,
            is_error=error,
            result=payload,
        )


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
