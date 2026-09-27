from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from fa import __version__
from fa.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("version", "config", "universe"):
        assert cmd in result.output
    assert "Not investment advice" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_check_ok(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "do-not-print-me")
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 0, result.output
    assert "Config OK" in result.output
    assert "FRED_API_KEY" in result.output
    assert "do-not-print-me" not in result.output


def test_config_check_fails_on_bad_config(config_copy: Path) -> None:
    path = config_copy / "forecasting.yaml"
    data = yaml.safe_load(path.read_text())
    data["horizons"] = [20, 5]
    path.write_text(yaml.safe_dump(data))
    result = runner.invoke(app, ["config", "check", "--config-dir", str(config_copy)])
    assert result.exit_code == 1


def test_universe_lists_wti() -> None:
    result = runner.invoke(app, ["universe"])
    assert result.exit_code == 0
    assert "CL=F" in result.output
    assert "nymex_cl" in result.output


def test_doctor_offline(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = CliRunner().invoke(app, ["doctor", "--no-network"])
    assert res.exit_code == 0, res.output
    assert "config valid" in res.output and "cache writable" in res.output
