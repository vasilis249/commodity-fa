"""`fa` command-line interface. Thin layer: parse args, call services, print results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from fa import __version__
from fa.config import DEFAULT_CONFIG_DIR, AppConfig, load_config, load_secrets
from fa.data.cli import app as data_app

DISCLAIMER = "Research tool, paper trading only. Not investment advice."

app = typer.Typer(
    name="fa",
    help=f"Energy-commodity analysis and probabilistic forecasting. {DISCLAIMER}",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect and validate config/*.yaml.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")

ConfigDirOpt = Annotated[
    Path, typer.Option("--config-dir", help="Directory holding the YAML config files.")
]


def _load_or_exit(config_dir: Path) -> AppConfig:
    try:
        return load_config(config_dir)
    except (ValidationError, ValueError, OSError) as exc:
        typer.secho(f"Config invalid: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@config_app.command("check")
def config_check(config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR) -> None:
    """Validate every config file and show which API keys are set (never their values)."""
    cfg = _load_or_exit(config_dir)
    typer.secho(f"Config OK ({cfg.config_dir})", fg=typer.colors.GREEN)
    typer.echo(f"  universe:       {len(cfg.data.universe)} instruments")
    typer.echo(f"  horizons:       {cfg.forecasting.horizons} trading days")
    enabled = [name for name, m in cfg.forecasting.models.items() if m.enabled]
    typer.echo(f"  models:         {', '.join(enabled)}")
    typer.echo(f"  debate rounds:  {cfg.agents.debate_rounds}")
    typer.echo(f"  budget:         ${cfg.models.budget.max_usd_per_run:.2f} per run")
    typer.echo("API keys:")
    for name, is_set in load_secrets().status().items():
        mark = typer.style("set", fg=typer.colors.GREEN) if is_set else "missing"
        typer.echo(f"  {name:<18} {mark}")


@app.command()
def analyze(
    symbol: str,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Numeric snapshot only (the only mode until Phase 4).")
    ] = False,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Decision date YYYY-MM-DD (point in time).")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use cached data only.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the snapshot as JSON.")] = False,
    curve: Annotated[bool, typer.Option("--curve/--no-curve", help="Fetch the live curve.")] = True,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Numeric snapshot: price, technicals, risk, seasonality, inventories, COT, curve."""
    from datetime import date

    from fa.analytics.snapshot import build_snapshot
    from fa.data.http import DataUnavailable
    from fa.data.service import DataService
    from fa.reports.snapshot_text import render

    cfg = _load_or_exit(config_dir)
    try:
        snap = build_snapshot(
            DataService(cfg, offline=offline),
            symbol,
            as_of=date.fromisoformat(as_of) if as_of else None,
            include_curve=curve,
        )
    except DataUnavailable as exc:
        typer.secho(f"Data unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(snap.model_dump_json(indent=2))
        return
    typer.echo(render(snap))
    if not no_llm:
        typer.secho("\n(LLM analysis arrives in Phase 4; this is the numeric snapshot.)", dim=True)


@app.command()
def forecast(
    symbol: str,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated subset, e.g. naive,lightgbm.")
    ] = None,
    folds: Annotated[int | None, typer.Option("--folds", help="Override max folds.")] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Forecast origin YYYY-MM-DD (point in time).")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use cached data only.")] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-run the walk-forward evaluation.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Probabilistic 1/5/20-day forecast with the models' measured out-of-sample skill."""
    from datetime import date

    from fa.data.http import DataUnavailable
    from fa.data.service import DataService
    from fa.forecasting.service import run_forecast
    from fa.reports.forecast_text import render

    cfg = _load_or_exit(config_dir)
    if folds:
        wf = cfg.forecasting.walk_forward.model_copy(update={"max_folds": folds})
        fc = cfg.forecasting.model_copy(update={"walk_forward": wf})
        cfg = cfg.model_copy(update={"forecasting": fc})
    wanted = [m.strip() for m in models.split(",")] if models else None
    if not as_json:
        typer.secho(
            "Running walk-forward evaluation (cached after the first run)...", dim=True, err=True
        )
    try:
        report = run_forecast(
            DataService(cfg, offline=offline),
            symbol,
            models=wanted,
            as_of=date.fromisoformat(as_of) if as_of else None,
            refresh=refresh,
        )
    except (DataUnavailable, ValueError) as exc:
        typer.secho(f"Forecast unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.model_dump_json(indent=2) if as_json else render(report))


@app.command()
def report(
    symbol: str,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Decision date YYYY-MM-DD (point in time).")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use cached data only.")] = False,
    rounds: Annotated[
        int | None, typer.Option("--rounds", help="Bull/bear debate rounds (config default).")
    ] = None,
    budget: Annotated[
        float | None, typer.Option("--budget", help="Override the USD budget for this run.")
    ] = None,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Full agent report: analysts, bull/bear debate, risk review, rating (computed in code)."""
    from datetime import date

    from fa.agents.base import AnthropicLLM
    from fa.config import load_secrets
    from fa.data.http import DataUnavailable
    from fa.data.service import DataService
    from fa.orchestration.pipeline import PipelineAborted, run_report
    from fa.reports.render import save

    cfg = _load_or_exit(config_dir)
    if budget is not None:
        b = cfg.models.budget.model_copy(update={"max_usd_per_run": budget})
        cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"budget": b})})
    if not load_secrets().anthropic_api_key:
        typer.secho("ANTHROPIC_API_KEY is not set (add it to .env).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.secho(
        f"Running agents (budget ${cfg.models.budget.max_usd_per_run:.2f})...", dim=True, err=True
    )
    try:
        rep = run_report(
            cfg,
            DataService(cfg, offline=offline),
            symbol,
            AnthropicLLM(),
            as_of=date.fromisoformat(as_of) if as_of else None,
            debate_rounds=rounds,
        )
    except PipelineAborted as exc:
        typer.secho(
            f"Aborted: {exc} (spent ${exc.spent_usd:.3f}); scratchpad: {exc.scratchpad}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except DataUnavailable as exc:
        typer.secho(f"Data unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    paths = save(rep, cfg.config_dir.parent / "reports")
    typer.secho(f"{rep.symbol}: {rep.rating} (conviction {rep.conviction:.2f})", bold=True)
    typer.echo(rep.thesis)
    typer.echo(f"Suggested max exposure: {rep.max_exposure:.0%}")
    v = rep.validation
    typer.echo(
        f"Validation: {v.numeric_violations_fixed} numeric fixes, "
        f"{len(v.sentences_removed)} sentences removed, "
        f"{len(v.unresolved_validator_issues)} unresolved validator issues"
    )
    typer.echo(
        f"Run cost: ${rep.cost.usd:.3f} (budget ${rep.cost.budget_usd:.2f}), "
        f"{rep.cost.tokens:,} tokens, {rep.cost.calls} calls"
    )
    typer.echo(f"Report: {paths['md']} (+ .json, .html)\nScratchpad: {rep.scratchpad}")
    typer.secho(rep.disclaimer, dim=True)


@app.command()
def backtest(
    symbol: str,
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="sma | forecast | agent")
    ] = "sma",
    offline: Annotated[bool, typer.Option("--offline", help="Use cached data only.")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm an agent backtest estimated above budget.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Paper backtest of a strategy vs buy-and-hold, with costs (paper only)."""
    from fa.backtest.agent_signal import BacktestCostWarning
    from fa.backtest.service import run_strategy
    from fa.config import load_secrets
    from fa.data.http import DataUnavailable
    from fa.data.service import DataService

    cfg = _load_or_exit(config_dir)
    if strategy not in ("sma", "forecast", "agent"):
        typer.secho("strategy must be sma, forecast or agent", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    kw: dict[str, object] = {}
    if strategy == "agent":
        if not load_secrets().anthropic_api_key:
            typer.secho(
                "ANTHROPIC_API_KEY is not set (add it to .env).", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=2)
        kw["confirm"] = yes
    try:
        rep = run_strategy(DataService(cfg, offline=offline), symbol, strategy, **kw)  # type: ignore[arg-type]
    except BacktestCostWarning as exc:
        typer.secho(f"{exc}. Re-run with --yes to proceed.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=3) from exc
    except (DataUnavailable, ValueError) as exc:
        typer.secho(f"Backtest failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(rep.model_dump_json(indent=2))
        return
    from fa.reports.backtest_text import render

    typer.echo(render(rep))


@app.command()
def universe(config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR) -> None:
    """List the configured instruments."""
    cfg = _load_or_exit(config_dir)
    typer.echo(f"{'SYMBOL':<8} {'CLASS':<10} {'UNIT':<10} {'ROLL':<26} {'COT':<7} NAME")
    for i in cfg.data.universe:
        cot = i.cot_market_code or "-"
        typer.echo(
            f"{i.symbol:<8} {i.asset_class:<10} {i.unit:<10} {i.roll_rule:<26} {cot:<7} {i.name}"
        )


if __name__ == "__main__":
    app()
