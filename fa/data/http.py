"""Polite HTTP: one shared session, per-provider rate limit, retry with backoff."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

import requests

from fa import __version__
from fa.config import ProviderLimits

log = logging.getLogger(__name__)

USER_AGENT = f"commodity-fa/{__version__} (personal research tool)"
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_SECRET_PARAM = re.compile(r"(api_key|apikey|token|key)=[^&\s'\")]+", re.IGNORECASE)
_SECRET_FIELD = re.compile(
    r"([\"']?(?:api_key|apikey|token|secret|password)[\"']?\s*:\s*[\"']?)[^\"',}\s]+",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    """Hide secrets in URL query strings and in dict/JSON text (errors often echo both)."""
    return _SECRET_FIELD.sub(r"\1***", _SECRET_PARAM.sub(r"\1=***", text))


class DataUnavailable(RuntimeError):
    """A provider could not deliver the data (after retries) or returned nothing."""


class TransientError(Exception):
    """A failure worth retrying (rate limit, 5xx, flaky upstream)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic):
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._last = -float("inf")

    def wait(self) -> None:
        delay = self._last + self.min_interval_s - self._clock()
        if delay > 0:
            time.sleep(delay)
        self._last = self._clock()


def with_retries[T](
    fn: Callable[[], T],
    limits: ProviderLimits,
    limiter: RateLimiter,
    what: str,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` under the rate limit, retrying transient failures with backoff."""
    last_exc: Exception | None = None
    for attempt in range(limits.max_retries + 1):
        limiter.wait()
        try:
            return fn()
        except TransientError as exc:
            last_exc = exc
            if exc.retry_after and exc.retry_after > max_delay_s:
                # e.g. a daily quota: waiting it out would hang the caller
                raise DataUnavailable(
                    f"{what}: rate limited, retry after {exc.retry_after:.0f}s"
                ) from exc
            delay = exc.retry_after or base_delay_s * 2**attempt
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            delay = base_delay_s * 2**attempt
        if attempt < limits.max_retries:
            log.warning(
                "%s failed (%s); retry %d in %.0fs", what, redact(str(last_exc)), attempt + 1, delay
            )
            sleep(delay)
    raise DataUnavailable(
        f"{what}: gave up after {limits.max_retries + 1} attempts ({redact(str(last_exc))})"
    ) from None  # the chained exception would print the raw URL, key included


class HttpClient:
    """GET with rate limiting and retries. Raises DataUnavailable on permanent failure."""

    def __init__(self, limits: ProviderLimits, name: str, timeout_s: float = 30.0):
        self.limits = limits
        self.name = name
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(limits.min_interval_s)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.requests_made = 0

    def get(self, url: str, params: dict[str, object] | None = None) -> requests.Response:
        def once() -> requests.Response:
            self.requests_made += 1
            resp = self.session.get(url, params=params, timeout=self.timeout_s)  # type: ignore[arg-type]
            if resp.status_code in RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                raise TransientError(
                    f"HTTP {resp.status_code}",
                    float(retry_after) if retry_after and retry_after.isdigit() else None,
                )
            if resp.status_code >= 400:
                raise DataUnavailable(
                    f"{self.name}: HTTP {resp.status_code} for {redact(str(resp.url))}"
                )
            return resp

        return with_retries(once, self.limits, self.limiter, what=f"{self.name} GET")
