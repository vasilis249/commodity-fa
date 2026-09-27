"""Per-run LLM cost tracking with a hard budget (config/models.yaml: budget).

Two gates:
- before a call: the estimated cost (input estimate + a fraction of max_tokens) must fit;
- after a call: the actual cost is charged; going over raises BudgetExceeded.
The pipeline catches BudgetExceeded, logs it and aborts cleanly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from fa.config import Budget, Price


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage: object) -> Usage:
        def get(name: str) -> int:
            return int(getattr(usage, name, 0) or 0)

        return cls(
            get("input_tokens"),
            get("output_tokens"),
            get("cache_creation_input_tokens"),
            get("cache_read_input_tokens"),
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def cost_usd(price: Price, usage: Usage) -> float:
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1_000_000


@dataclass
class CostTracker:
    pricing: dict[str, Price]
    budget: Budget
    spent_usd: float = 0.0
    tokens: int = 0
    calls: int = 0
    by_agent: dict[str, float] = field(default_factory=dict)
    reserved_usd: float = 0.0  # estimates of calls in flight (parallel agents)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _price(self, model: str) -> Price:
        if model not in self.pricing:
            raise KeyError(f"no pricing for model {model!r} in config/models.yaml")
        return self.pricing[model]

    def preflight(self, model: str, est_input_tokens: int, max_tokens: int) -> float:
        """Reserve this call's estimated cost; raise if it could push the run over budget.

        Input is priced at the cache-write rate (the first write of a cached prefix costs
        1.25x). Reservations are held under the lock, so parallel agents cannot all pass
        the check before any of them is charged. Settle with `charge(..., reserved=est)`.
        """
        est_out = int(max_tokens * self.budget.preflight_output_fraction)
        price = self._price(model)
        est = cost_usd(price, Usage(0, est_out, est_input_tokens, 0))
        with self._lock:
            committed = self.spent_usd + self.reserved_usd
            if committed + est > self.budget.max_usd_per_run:
                raise BudgetExceeded(
                    f"next call (~${est:.3f}) would exceed the ${self.budget.max_usd_per_run:.2f} "
                    f"budget (spent ${self.spent_usd:.3f}, in flight ${self.reserved_usd:.3f})"
                )
            if self.tokens + est_input_tokens + est_out > self.budget.max_tokens_per_run:
                raise BudgetExceeded(
                    f"token budget {self.budget.max_tokens_per_run} would be exceeded"
                )
            self.reserved_usd += est
        return est

    def release(self, reserved: float) -> None:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)

    def charge(self, agent: str, model: str, usage: Usage, reserved: float = 0.0) -> float:
        cost = cost_usd(self._price(model), usage)
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            self.spent_usd += cost
            self.tokens += usage.total
            self.calls += 1
            self.by_agent[agent] = self.by_agent.get(agent, 0.0) + cost
        if self.spent_usd > self.budget.max_usd_per_run:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.3f} > ${self.budget.max_usd_per_run:.2f} budget"
            )
        return cost
