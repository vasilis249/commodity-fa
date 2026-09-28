from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fa.config import AppConfig, RollRule, Secrets, load_config


def _edit(config_dir: Path, name: str, **changes: object) -> None:
    path = config_dir / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
    data.update(changes)
    path.write_text(yaml.safe_dump(data))


def test_repo_config_is_valid(cfg: AppConfig) -> None:
    assert cfg.data.universe
    assert cfg.forecasting.horizons == [1, 5, 20]
    assert cfg.forecasting.quantiles == [0.1, 0.5, 0.9]
    assert cfg.models.budget.max_usd_per_run > 0


def test_every_futures_contract_has_a_roll_rule(cfg: AppConfig) -> None:
    for inst in cfg.data.universe:
        if inst.asset_class == "commodity":
            assert inst.roll_rule is not RollRule.NONE, inst.symbol
        else:
            assert inst.roll_rule is RollRule.NONE, inst.symbol


def test_energy_core_is_in_universe(cfg: AppConfig) -> None:
    for symbol in ("CL=F", "BZ=F", "NG=F", "HO=F", "RB=F"):
        assert cfg.data.instrument(symbol) is not None, symbol


def test_every_agent_has_a_model_role(cfg: AppConfig) -> None:
    assert set(cfg.agents.agents) <= set(cfg.models.agent_roles)


def test_timesfm_off_by_default(cfg: AppConfig) -> None:
    # Latest TimesFM weights are non-commercial; must be an explicit opt-in.
    assert not cfg.forecasting.models["timesfm"].enabled


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("forecasting", {"horizons": [5, 1, 20]}),
        ("forecasting", {"horizons": [0, 5]}),
        ("forecasting", {"quantiles": [0.1, 1.2]}),
        ("forecasting", {"quantiles": [0.9, 0.1]}),
        (
            "models",
            {"budget": {"max_usd_per_run": 0, "max_usd_backtest": 1, "max_tokens_per_run": 1}},
        ),
        ("agents", {"decision_weights": {"a": 0.5, "b": 0.6}}),
        ("risk", {"max_gross_exposure": 2.0}),
        ("data", {"unexpected_key": 1}),
    ],
)
def test_invalid_config_is_rejected(
    config_copy: Path, name: str, changes: dict[str, object]
) -> None:
    _edit(config_copy, name, **changes)
    with pytest.raises(ValidationError):
        load_config(config_copy)


def test_embargo_must_cover_roll_mask_hindsight(config_copy: Path) -> None:
    data = yaml.safe_load((config_copy / "forecasting.yaml").read_text())
    data["walk_forward"]["embargo_days"] = 2  # < window_before + window_after = 5
    (config_copy / "forecasting.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="embargo_days"):
        load_config(config_copy)


def test_futures_without_roll_rule_is_rejected(config_copy: Path) -> None:
    data = yaml.safe_load((config_copy / "data.yaml").read_text())
    data["universe"][0]["roll_rule"] = "none"
    (config_copy / "data.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="roll_rule"):
        load_config(config_copy)


def test_role_with_unpriced_model_is_rejected(config_copy: Path) -> None:
    data = yaml.safe_load((config_copy / "models.yaml").read_text())
    data["roles"]["analyst"]["model"] = "claude-unknown"
    (config_copy / "models.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="no pricing"):
        load_config(config_copy)


def test_secret_status_never_exposes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "super-secret-value")
    secrets = Secrets(_env_file=None)
    status = secrets.status()
    assert status["EIA_API_KEY"] is True
    assert "super-secret-value" not in repr(secrets)
    assert "super-secret-value" not in str(status)


def test_blank_keys_in_env_file_count_as_missing(tmp_path, monkeypatch) -> None:
    from fa.config import Secrets

    for k in ("ANTHROPIC_API_KEY", "EIA_API_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "ANTHROPIC_API_KEY=\nEIA_API_KEY=abc123\nFRED_API_KEY=   \n"
    )  # .env.example style
    s = Secrets(_env_file=env)
    assert s.status() == {"ANTHROPIC_API_KEY": False, "EIA_API_KEY": True, "FRED_API_KEY": False}


def test_anthropic_client_gets_the_key_from_env_file(tmp_path, monkeypatch) -> None:
    """The SDK reads only os.environ; a key kept in .env must still reach it."""
    from fa.agents.base import AnthropicLLM
    from fa.config import Secrets

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")
    monkeypatch.setattr("fa.config.Secrets.model_config", {**Secrets.model_config, "env_file": env})
    assert AnthropicLLM().client.api_key == "sk-ant-from-dotenv"
