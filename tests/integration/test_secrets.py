"""Guardrail: no secrets in code, logs or reports.

Fake keys are set in the environment, every artifact type is produced (report,
scratchpads, research answer, eval results, backtest, CLI output, provider errors and
retry logs), and every byte written is scanned for the key strings.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest
import requests
from typer.testing import CliRunner

from fa import evals
from fa.backtest.service import run_strategy
from fa.config import AppConfig, Secrets
from fa.data.http import DataUnavailable
from fa.forecasting.service import forecast_dataset
from fa.orchestration.pipeline import run_report
from fa.orchestration.research_loop import ask
from fa.reports.render import save
from tests.agent_helpers import FakeLLM, offline_service
from tests.forecast_helpers import small_config, synthetic_dataset

SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-TESTSECRET-0123456789abcdef",
    "EIA_API_KEY": "eiaTESTSECRET9876543210",
    "FRED_API_KEY": "fredTESTSECRET55555",
}


@pytest.fixture
def keys(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    assert Secrets(_env_file=None).status() == dict.fromkeys(SECRETS, True)


def _leaks(text: str) -> list[str]:
    return [k for k, v in SECRETS.items() if v in text or v[-12:] in text]


def _scan(root: Path) -> dict[str, list[str]]:
    found = {}
    for p in root.rglob("*"):
        if p.is_file():
            hits = _leaks(p.read_bytes().decode("utf-8", errors="ignore"))
            if hits:
                found[str(p)] = hits
    return found


def test_no_secret_in_any_artifact(cfg: AppConfig, tmp_path: Path, monkeypatch, keys) -> None:
    small = small_config(cfg)
    fc = small.forecasting.model_copy(update={"leaderboard_dir": tmp_path / "lb"})
    bt = small.backtest.model_copy(update={"results_dir": tmp_path / "bt"})
    small = small.model_copy(update={"forecasting": fc, "backtest": bt})
    ds = synthetic_dataset(small, n=700)
    monkeypatch.setattr(
        "fa.tools.market.run_forecast",
        lambda svc, symbol, as_of=None: forecast_dataset(ds, small, models=["naive", "drift"]),
    )
    svc = offline_service(small, tmp_path)
    rep = run_report(small, svc, "CL=F", FakeLLM(), runs_dir=tmp_path / "runs", parallel=False)
    save(rep, tmp_path / "reports")
    ask(
        small,
        svc,
        "WTI settle?",
        "CL=F",
        FakeLLM(),
        as_of=date(2026, 6, 1),
        runs_dir=tmp_path / "runs",
    )
    q = evals.Question(
        id="p",
        category="price",
        symbol="CL=F",
        as_of=date(2026, 6, 1),
        question="WTI?",
        expect=evals.Expect(type="number", value=1.0),
        ref=evals.Ref(kind="field", tool="price_technicals", path="price.last_close"),
    )
    evals.save(
        evals.run_evals(small, svc, [q], solver="llm", llm=FakeLLM(), runs_dir=tmp_path / "runs"),
        tmp_path / "evals",
    )
    run_strategy(svc, "CL=F", "sma", ds=None)
    assert list((tmp_path / "runs").glob("*.jsonl")) and list((tmp_path / "bt").glob("*.json"))
    assert _scan(tmp_path) == {}


def test_cli_never_prints_keys(keys) -> None:
    from fa.cli import app

    out = CliRunner().invoke(app, ["config", "check"]).output
    assert "set" in out and not _leaks(out)


def test_provider_errors_and_retry_logs_are_redacted(monkeypatch, caplog, keys) -> None:
    from fa.config import load_config
    from fa.data.providers.eia import EIAProvider

    limits = load_config().data.providers["eia"].model_copy(update={"max_retries": 1})
    prov = EIAProvider(limits, SECRETS["EIA_API_KEY"])
    url = f"https://api.eia.gov/v2/seriesid/X?api_key={SECRETS['EIA_API_KEY']}"

    def forbidden(*a, **k):
        r = requests.Response()
        r.status_code, r.url = 403, url
        return r

    monkeypatch.setattr(prov.http.session, "get", forbidden)
    with pytest.raises(DataUnavailable) as err:
        prov.http.get("https://api.eia.gov/v2/seriesid/X", {"api_key": SECRETS["EIA_API_KEY"]})
    assert not _leaks(str(err.value)) and "***" in str(err.value)

    def down(*a, **k):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr(prov.http.session, "get", down)
    monkeypatch.setattr("fa.data.http.time.sleep", lambda s: None)
    with caplog.at_level(logging.WARNING), pytest.raises(DataUnavailable) as err2:
        prov.http.get("https://api.eia.gov/v2/seriesid/X", {"api_key": SECRETS["EIA_API_KEY"]})
    assert not _leaks(str(err2.value)) and not _leaks(caplog.text)
    assert err2.value.__cause__ is None  # the raw exception (with the URL) is not chained
