"""`fa data ...` commands: fetch, inspect and query cached data."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from fa.config import DEFAULT_CONFIG_DIR, load_config
from fa.data.http import DataUnavailable
from fa.data.quality import QualityReport, Severity
from fa.data.service import DataService, FetchInfo

app = typer.Typer(help="Fetch, check and query market data (cached on disk).", no_args_is_help=True)

ConfigDirOpt = Annotated[Path, typer.Option("--config-dir", help="Config directory.")]
RefreshOpt = Annotated[bool, typer.Option("--refresh", help="Ignore the cache and refetch.")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="Use the cache only; no network.")]

SEVERITY_COLOR = {
    Severity.INFO: None,
    Severity.WARN: typer.colors.YELLOW,
    Severity.ERROR: typer.colors.RED,
}


def _service(config_dir: Path, offline: bool) -> DataService:
    return DataService(load_config(config_dir), offline=offline)


def _fail(exc: Exception) -> None:
    typer.secho(f"Data unavailable: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _print_info(info: FetchInfo) -> None:
    color = typer.colors.GREEN if info.source == "cache" else typer.colors.CYAN
    stale = typer.style("  (stale)", fg=typer.colors.YELLOW) if info.stale else ""
    typer.echo(
        f"source: {typer.style(info.source, fg=color)}  network requests: {info.requests}  "
        f"cached rows: {info.rows}  last fetch: {info.fetched_at:%Y-%m-%d %H:%M} UTC{stale}"
    )


def _print_quality(report: QualityReport, verbose: bool) -> None:
    status = (
        typer.style("OK", fg=typer.colors.GREEN)
        if report.ok
        else typer.style("ERRORS", fg=typer.colors.RED)
    )
    typer.echo(f"quality: {status}  ({report.rows} rows, {report.start} .. {report.end})")
    for issue in report.issues:
        if issue.severity is Severity.INFO and not verbose:
            continue
        tag = typer.style(f"[{issue.severity}]", fg=SEVERITY_COLOR[issue.severity])
        first = f"  first: {issue.first}" if issue.first else ""
        typer.echo(f"  {tag} {issue.check}: {issue.detail}{first}")


def _show(frame: pd.DataFrame, n: int) -> None:
    if n > 0:
        with pd.option_context("display.width", 140, "display.max_columns", 20):
            typer.echo(frame.tail(n).to_string())


@app.command()
def prices(
    symbol: str,
    years: Annotated[float, typer.Option(help="Years of history.")] = 10,
    tail: Annotated[int, typer.Option(help="Rows to print.")] = 5,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show info-level checks.")
    ] = False,
    refresh: RefreshOpt = False,
    offline: OfflineOpt = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Daily prices with roll-adjusted returns and a data-quality report."""
    try:
        data = _service(config_dir, offline).prices(symbol, years=years, refresh=refresh)
    except DataUnavailable as exc:
        _fail(exc)
        return
    inst = data.instrument
    typer.secho(f"{inst.symbol}  {inst.name}  [{inst.unit}]", bold=True)
    _print_info(data.info)
    masked = data.frame["mask_reason"].value_counts().drop("", errors="ignore")
    typer.echo("masked returns: " + (", ".join(f"{k}={v}" for k, v in masked.items()) or "none"))
    _print_quality(data.quality, verbose)
    _show(
        data.frame[["open", "high", "low", "close", "volume", "ret", "mask_reason", "close_adj"]],
        tail,
    )


@app.command()
def eia(
    symbol: str,
    tail: Annotated[int, typer.Option(help="Rows to print per series.")] = 3,
    refresh: RefreshOpt = False,
    offline: OfflineOpt = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """EIA weekly series configured for an instrument (with release timestamps)."""
    svc = _service(config_dir, offline)
    inst = svc.instrument(symbol)
    if not inst.eia_series:
        typer.echo(f"{symbol}: no EIA series configured")
        return
    for name, sid in inst.eia_series.items():
        try:
            frame, info = svc.eia(sid, refresh=refresh)
        except DataUnavailable as exc:
            _fail(exc)
            return
        typer.secho(f"{name} ({sid})", bold=True)
        _print_info(info)
        _show(frame, tail)


@app.command()
def cot(
    symbol: str,
    tail: Annotated[int, typer.Option(help="Rows to print.")] = 5,
    refresh: RefreshOpt = False,
    offline: OfflineOpt = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """CFTC Commitments of Traders positions (disaggregated, futures only)."""
    try:
        frame, info = _service(config_dir, offline).cot(symbol, refresh=refresh)
    except DataUnavailable as exc:
        _fail(exc)
        return
    _print_info(info)
    _show(
        frame[["open_interest", "mm_long", "mm_short", "prod_long", "prod_short", "available_at"]],
        tail,
    )


@app.command()
def fred(
    series_id: str,
    tail: Annotated[int, typer.Option(help="Rows to print.")] = 5,
    refresh: RefreshOpt = False,
    offline: OfflineOpt = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """A FRED series, first-release values, with availability timestamps."""
    try:
        frame, info = _service(config_dir, offline).fred(series_id, refresh=refresh)
    except DataUnavailable as exc:
        _fail(exc)
        return
    _print_info(info)
    _show(frame, tail)


@app.command()
def news(
    symbol: str,
    limit: Annotated[int, typer.Option(help="Headlines to print.")] = 15,
    refresh: RefreshOpt = False,
    offline: OfflineOpt = False,
    config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR,
) -> None:
    """Recent headlines for an instrument (live only; not usable in backtests)."""
    try:
        frame, infos = _service(config_dir, offline).news(symbol, refresh=refresh)
    except DataUnavailable as exc:
        _fail(exc)
        return
    net = sum(i.requests for i in infos)
    typer.echo(f"{len(frame)} headlines from {len(infos)} feeds; network requests: {net}")
    for row in frame.head(limit).itertuples():
        typer.echo(
            f"  {row.published_at:%Y-%m-%d %H:%M}  {row.title}"
            + (f"  — {row.source}" if row.source else "")
        )


@app.command("cache")
def cache_list(config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR) -> None:
    """List cache entries."""
    svc = _service(config_dir, offline=True)
    entries = svc.cache.entries()
    if not entries:
        typer.echo(f"cache is empty ({svc.cache.root})")
        return
    typer.echo(f"{'PROVIDER':<8} {'ROWS':>6}  {'FETCHED (UTC)':<16}  KEY")
    for m in entries:
        typer.echo(f"{m.provider:<8} {m.rows:>6}  {m.fetched_at:%Y-%m-%d %H:%M}  {m.key}")


@app.command()
def sql(query: str, config_dir: ConfigDirOpt = DEFAULT_CONFIG_DIR) -> None:
    """Run DuckDB SQL over cached files; each provider is a view (yahoo, eia, cftc, fred, news)."""
    svc = _service(config_dir, offline=True)
    with pd.option_context("display.width", 140, "display.max_columns", 20, "display.max_rows", 50):
        typer.echo(svc.cache.query(query).to_string())
