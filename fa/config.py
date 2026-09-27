"""Load and validate config/*.yaml and secrets from .env.

Every YAML file maps to a Pydantic model; invalid config fails loudly at load time.
Secrets are read by pydantic-settings and never printed or logged.
"""

from __future__ import annotations

from datetime import date, time
from enum import StrEnum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- data.yaml -----------------------------------------------------------------------


class RollRule(StrEnum):
    """How a continuous futures series rolls; implemented in the data layer (Phase 1)."""

    NYMEX_CL = "nymex_cl"
    NYMEX_NG = "nymex_ng"
    NYMEX_LAST_BD_PRIOR_MONTH = "nymex_last_bd_prior_month"
    ICE_BRENT = "ice_brent"
    ICE_ENDEX_TTF = "ice_endex_ttf"
    NONE = "none"


class Instrument(_Strict):
    symbol: str
    name: str
    asset_class: Literal["commodity", "etf", "equity", "index"]
    sector: str
    exchange: str
    currency: str
    unit: str
    roll_rule: RollRule
    settle_time: time  # daily settlement/close, local to `timezone`
    timezone: str
    contract_root: str | None = None  # e.g. "CL"; individual contracts are CLZ26 etc.
    contract_suffix: str | None = None  # Yahoo suffix for single contracts, e.g. ".NYM"
    cot_market_code: str | None = None
    eia_series: dict[str, str] = Field(default_factory=dict)
    fred_spot: str | None = None
    news_queries: list[str] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, v: str) -> str:
        ZoneInfo(v)  # raises for unknown zones
        return v

    @model_validator(mode="after")
    def _roll_rule_matches_class(self) -> Instrument:
        if self.asset_class == "commodity" and self.roll_rule is RollRule.NONE:
            raise ValueError(f"{self.symbol}: futures need a roll_rule other than 'none'")
        if self.asset_class != "commodity" and self.roll_rule is not RollRule.NONE:
            raise ValueError(f"{self.symbol}: only commodity futures take a roll_rule")
        return self


Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday"]


class ReleaseSchedule(_Strict):
    """A weekly publication: the first `weekday` after the period end, at `release_time` in `tz`.

    Each `calendar` holiday between the Monday of the release week and the nominal
    release day pushes the release one business day later (a conservative rule:
    a later timestamp never leaks).
    """

    weekday: Weekday
    release_time: time
    tz: str
    calendar: Literal["us_federal", "cme", "uk"]


class ReleaseOverride(_Strict):
    """Reports whose period falls in [period_start, period_end] were published late.

    Used for publisher disruptions (e.g. government shutdowns). `available_from` must be
    a conservative (late) bound: an early date leaks, a late one only loses freshness.
    """

    release: str
    period_start: date
    period_end: date
    available_from: date
    note: str

    @model_validator(mode="after")
    def _ordered(self) -> ReleaseOverride:
        if not self.period_start <= self.period_end < self.available_from:
            raise ValueError("need period_start <= period_end < available_from")
        return self


class RollSettings(_Strict):
    window_before: int = Field(ge=0)  # sessions before expiry in which a switch may happen
    window_after: int = Field(ge=0)
    min_volume_ratio: float = Field(gt=1)  # day-over-day volume jump marking the switch
    mask_after: int = Field(ge=0)  # extra sessions masked after the volume jump


class CacheTTL(_Strict):
    prices_hours: float = Field(gt=0)
    eia_hours: float = Field(gt=0)
    cot_hours: float = Field(gt=0)
    fred_hours: float = Field(gt=0)
    news_hours: float = Field(gt=0)


class QualitySettings(_Strict):
    stale_sessions: int = Field(gt=0)  # last bar older than this many sessions -> stale
    max_gap_sessions: int = Field(gt=0)  # consecutive missing sessions tolerated
    repeated_close_run: int = Field(ge=2)  # identical closes in a row -> stale quote
    outlier_mad_multiple: float = Field(gt=0)


class ProviderLimits(_Strict):
    min_interval_s: float = Field(ge=0)
    max_retries: int = Field(ge=0)


REQUIRED_PROVIDERS = frozenset({"yahoo", "eia", "cftc", "fred", "news_rss"})


class DataConfig(_Strict):
    cache_dir: Path
    default_history_years: int = Field(gt=0)
    universe: list[Instrument]
    fred_macro: dict[str, str]
    fred_availability_lag_days: int = Field(ge=0)
    releases: dict[str, ReleaseSchedule]
    release_overrides: list[ReleaseOverride] = Field(default_factory=list)
    eia_release_by_prefix: dict[str, str]
    rolls: RollSettings
    quality: QualitySettings
    cache_ttl: CacheTTL
    news_feeds: list[str]
    curve_contracts: int = Field(ge=2, le=24)
    providers: dict[str, ProviderLimits]

    @model_validator(mode="after")
    def _references_resolve(self) -> DataConfig:
        for prefix, release in self.eia_release_by_prefix.items():
            if release not in self.releases:
                raise ValueError(f"eia_release_by_prefix.{prefix} -> unknown release '{release}'")
        for inst in self.universe:
            for sid in inst.eia_series.values():
                if sid.split(".", 1)[0] not in self.eia_release_by_prefix:
                    raise ValueError(f"{inst.symbol}: no release schedule for EIA series {sid}")
        for ov in self.release_overrides:
            if ov.release not in self.releases:
                raise ValueError(f"release_overrides: unknown release '{ov.release}'")
        if "cftc_cot" not in self.releases:
            raise ValueError("releases.cftc_cot is required")
        missing = REQUIRED_PROVIDERS - set(self.providers)
        if missing:
            raise ValueError(f"providers missing rate limits: {sorted(missing)}")
        return self

    @field_validator("universe")
    @classmethod
    def _unique_symbols(cls, v: list[Instrument]) -> list[Instrument]:
        symbols = [i.symbol for i in v]
        dupes = {s for s in symbols if symbols.count(s) > 1}
        if dupes:
            raise ValueError(f"duplicate symbols in universe: {sorted(dupes)}")
        return v

    def instrument(self, symbol: str) -> Instrument | None:
        return next((i for i in self.universe if i.symbol == symbol), None)


# --- forecasting.yaml ----------------------------------------------------------------


class WalkForward(_Strict):
    min_train_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    # No purge_days: Dataset.until() drops every label that needs prices after the
    # training cut, which purges overlapping labels by construction.
    embargo_days: int = Field(ge=0)
    max_folds: int = Field(gt=0)
    eval_stride: int = Field(gt=0)
    refit_every: int = Field(ge=1)  # folds between refits; in between, the last fit is reused


class ModelToggle(_Strict):
    enabled: bool


class Skill(_Strict):
    baseline: str
    significance_alpha: float = Field(gt=0, lt=1)


class BaselineSettings(_Strict):
    lookback_days: int = Field(gt=0)
    min_days: int = Field(gt=0)


class LightGBMSettings(_Strict):
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_lambda: float = Field(ge=0)
    seed: int
    num_threads: int = Field(ge=1)  # results are deterministic for a fixed thread count


class ChronosSettings(_Strict):
    model_id: str
    context_length: int = Field(gt=16)
    device: str


class EnsembleSettings(_Strict):
    min_folds: int = Field(ge=1)
    window_folds: int = Field(ge=1)


class ForecastingConfig(_Strict):
    target: Literal["log_return"]
    horizons: list[int]
    quantiles: list[float]
    interval_coverage_target: float = Field(gt=0, lt=1)
    walk_forward: WalkForward
    models: dict[str, ModelToggle]
    baselines: BaselineSettings
    lightgbm: LightGBMSettings
    chronos2: ChronosSettings
    ensemble: EnsembleSettings
    skill: Skill
    leaderboard_dir: Path

    @field_validator("horizons")
    @classmethod
    def _horizons_ascending(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v) or v != sorted(set(v)):
            raise ValueError("horizons must be positive, unique and ascending")
        return v

    @field_validator("quantiles")
    @classmethod
    def _quantiles_valid(cls, v: list[float]) -> list[float]:
        if not v or any(not 0 < q < 1 for q in v) or v != sorted(set(v)):
            raise ValueError("quantiles must be unique, ascending and inside (0, 1)")
        return v

    @model_validator(mode="after")
    def _baseline_known(self) -> ForecastingConfig:
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include the median (0.5)")
        if self.skill.baseline not in self.models:
            raise ValueError(f"skill.baseline '{self.skill.baseline}' is not in models")
        return self


# --- agents.yaml ---------------------------------------------------------------------


class AgentSpec(_Strict):
    enabled: bool
    tools: list[str]


class RatingThresholds(_Strict):
    strong: float = Field(gt=0, le=1)
    weak: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> RatingThresholds:
        if self.weak >= self.strong:
            raise ValueError("rating_thresholds: weak must be < strong")
        return self


class AgentsConfig(_Strict):
    debate_rounds: int = Field(ge=0, le=5)
    validator_max_loops: int = Field(ge=0, le=5)
    agent_max_steps: int = Field(ge=2, le=30)
    research_loop_max_steps: int = Field(gt=0)
    agents: dict[str, AgentSpec]
    decision_weights: dict[str, float]
    rating_thresholds: RatingThresholds

    @field_validator("decision_weights")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in v.values()) or abs(sum(v.values()) - 1.0) > 1e-9:
            raise ValueError("decision_weights must be non-negative and sum to 1")
        return v


# --- models.yaml ---------------------------------------------------------------------


class RoleSpec(_Strict):
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int = Field(gt=0)


class Budget(_Strict):
    max_usd_per_run: float = Field(gt=0)
    max_usd_backtest: float = Field(gt=0)
    max_tokens_per_run: int = Field(gt=0)
    preflight_output_fraction: float = Field(gt=0, le=1)


class Price(_Strict):
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write: float = Field(ge=0)
    cache_read: float = Field(ge=0)


class ModelsConfig(_Strict):
    default_role: str
    roles: dict[str, RoleSpec]
    agent_roles: dict[str, str]
    budget: Budget
    prompt_caching: bool
    pricing: dict[str, Price]

    @model_validator(mode="after")
    def _references_resolve(self) -> ModelsConfig:
        if self.default_role not in self.roles:
            raise ValueError(f"default_role '{self.default_role}' is not a defined role")
        for agent, role in self.agent_roles.items():
            if role not in self.roles:
                raise ValueError(f"agent '{agent}' maps to unknown role '{role}'")
        for name, spec in self.roles.items():
            if spec.model not in self.pricing:
                raise ValueError(f"role '{name}' uses '{spec.model}', which has no pricing")
        return self


# --- risk.yaml -----------------------------------------------------------------------


class BacktestCosts(_Strict):
    cost_bps_per_side: float = Field(ge=0)
    slippage_bps_per_side: float = Field(ge=0)
    roll_cost_bps: float = Field(ge=0)
    execution: Literal["next_open"]


class RiskConfig(_Strict):
    max_gross_exposure: float = Field(gt=0, le=1.0)  # paper only, no leverage
    max_position_fraction: float = Field(gt=0, le=1.0)
    vol_target_annual: float = Field(gt=0)
    max_drawdown_stop: float = Field(gt=0, lt=1)
    min_history_days: int = Field(gt=0)
    backtest: BacktestCosts


# --- secrets -------------------------------------------------------------------------


class Secrets(BaseSettings):
    """API keys from the environment or .env. Never log these values."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    eia_api_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None

    def status(self) -> dict[str, bool]:
        """Which keys are set, without exposing any value."""
        return {name.upper(): getattr(self, name) is not None for name in type(self).model_fields}


# --- top level -----------------------------------------------------------------------


class AppConfig(_Strict):
    config_dir: Path
    data: DataConfig
    forecasting: ForecastingConfig
    agents: AgentsConfig
    models: ModelsConfig
    risk: RiskConfig

    @model_validator(mode="after")
    def _cross_file_checks(self) -> AppConfig:
        # labels use the volume-detected roll mask, which looks up to
        # window_before + window_after sessions ahead: the embargo must cover that
        hindsight = self.data.rolls.window_before + self.data.rolls.window_after
        if self.forecasting.walk_forward.embargo_days < hindsight:
            raise ValueError(
                f"walk_forward.embargo_days must be >= {hindsight} "
                "(roll-mask hindsight: rolls.window_before + rolls.window_after)"
            )
        missing = set(self.agents.agents) - set(self.models.agent_roles)
        if missing:
            raise ValueError(f"agents without a model role in models.yaml: {sorted(missing)}")
        return self


CONFIG_FILES: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "forecasting": ForecastingConfig,
    "agents": AgentsConfig,
    "models": ModelsConfig,
    "risk": RiskConfig,
}


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def load_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> AppConfig:
    """Load and validate every config file in `config_dir`."""
    config_dir = Path(config_dir)
    sections = {name: _read_yaml(config_dir / f"{name}.yaml") for name in CONFIG_FILES}
    return AppConfig.model_validate({"config_dir": config_dir, **sections})


def load_secrets() -> Secrets:
    """Read API keys from the environment / .env (kept separate from YAML config)."""
    return Secrets()
