from __future__ import annotations

import pytest
import requests

from fa.config import ProviderLimits
from fa.data.http import DataUnavailable, HttpClient, RateLimiter, TransientError, with_retries

LIMITS = ProviderLimits(min_interval_s=0, max_retries=3)


def test_with_retries_recovers_after_transient_errors() -> None:
    calls, sleeps = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("503")
        return "ok"

    out = with_retries(flaky, LIMITS, RateLimiter(0), "t", base_delay_s=1, sleep=sleeps.append)
    assert out == "ok"
    assert sleeps == [1, 2]  # exponential backoff


def test_with_retries_gives_up() -> None:
    def broken() -> None:
        raise requests.ConnectionError("down")

    with pytest.raises(DataUnavailable, match="4 attempts"):
        with_retries(broken, LIMITS, RateLimiter(0), "t", sleep=lambda s: None)


def test_long_retry_after_fails_fast_instead_of_hanging() -> None:
    def quota() -> None:
        raise TransientError("429", retry_after=3600)

    with pytest.raises(DataUnavailable, match="rate limited"):
        with_retries(quota, LIMITS, RateLimiter(0), "t", sleep=lambda s: pytest.fail("slept"))


class _Resp:
    def __init__(self, status: int, headers: dict[str, str] | None = None):
        self.status_code = status
        self.headers = headers or {}
        self.url = "https://example.test"


class _Session:
    def __init__(self, statuses: list[int]):
        self.statuses = statuses
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: object = None, timeout: float = 0) -> _Resp:
        return _Resp(self.statuses.pop(0), {"Retry-After": "0"})


def test_http_client_retries_429_and_stops_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fa.data.http.time.sleep", lambda s: None)
    client = HttpClient(LIMITS, "t")
    client.session = _Session([429, 200])  # type: ignore[assignment]
    assert client.get("u").status_code == 200
    assert client.requests_made == 2

    client.session = _Session([404])  # type: ignore[assignment]
    with pytest.raises(DataUnavailable, match="404"):
        client.get("u")
    assert client.requests_made == 3  # no retry on 404


def test_rate_limiter_spaces_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    slept: list[float] = []
    monkeypatch.setattr("fa.data.http.time.sleep", slept.append)
    limiter = RateLimiter(1.0, clock=lambda: now[0])
    limiter.wait()
    now[0] = 0.25
    limiter.wait()
    assert slept == [pytest.approx(0.75)]


def test_secrets_are_redacted_from_errors() -> None:
    from fa.data.http import redact

    url = "https://api.eia.gov/v2/seriesid/X?api_key=abc123SECRET&length=5"
    assert "abc123SECRET" not in redact(url)
    assert "length=5" in redact(url)

    def leaky() -> None:
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")

    with pytest.raises(DataUnavailable) as info:
        with_retries(leaky, LIMITS, RateLimiter(0), "eia GET", sleep=lambda s: None)
    assert "abc123SECRET" not in str(info.value)
    assert info.value.__cause__ is None
