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
