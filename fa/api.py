"""Service facade for user interfaces (Streamlit today, an HTTP API later).

UI-agnostic: no Streamlit imports, plain arguments in, Pydantic models or DataFrames out.
Every function takes a `Workspace` (config + where artifacts live + how to build the data
service), so tests and other front ends can inject their own.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from fa.config import DEFAULT_CONFIG_DIR, AppConfig, Instrument, load_config, load_secrets
from fa.data.service import DataService, PriceData

# --- workspace ------------------------------------------------------------------------


@dataclass
class Workspace:
    cfg: AppConfig
    service_factory: Callable[[bool], DataService] | None = None

    @classmethod
    def load(cls, config_dir: Path | None = None) -> Workspace:
        """Config from `config_dir`, else $FA_CONFIG_DIR, else the repo's config/. The
        workspace root (reports, runs, backtests, cache) is the config dir's parent."""
        env = os.environ.get("FA_CONFIG_DIR")
        return cls(load_config(config_dir or (Path(env) if env else DEFAULT_CONFIG_DIR)))

    def service(self, offline: bool = False) -> DataService:
        if self.service_factory is not None:
            return self.service_factory(offline)
        return DataService(self.cfg, offline=offline)

    @property
    def root(self) -> Path:
        return self.cfg.config_dir.parent

    def _dir(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.root / ".runs"

    @property
    def backtests_dir(self) -> Path:
        return self._dir(self.cfg.backtest.results_dir)

    @property
    def leaderboards_dir(self) -> Path:
        return self._dir(self.cfg.forecasting.leaderboard_dir)


def key_status() -> dict[str, bool]:
    """Which API keys are set (never their values)."""
    return load_secrets().status()


# --- instruments and data -------------------------------------------------------------


def search(ws: Workspace, query: str) -> list[Instrument]:
    """Configured instruments whose symbol, name, sector or unit match `query`."""
    q = query.strip().lower()
    if not q:
        return list(ws.cfg.data.universe)
    return [
        i
        for i in ws.cfg.data.universe
        if q in f"{i.symbol} {i.name} {i.sector} {i.unit} {i.asset_class}".lower()
    ]


def price_history(ws: Workspace, symbol: str, offline: bool = False) -> PriceData:
    return ws.service(offline).prices(symbol)


def snapshot(ws: Workspace, symbol: str, offline: bool = False, curve: bool = True) -> Any:
    from fa.analytics.snapshot import build_snapshot

    return build_snapshot(ws.service(offline), symbol, include_curve=curve)


def forecast(
    ws: Workspace,
    symbol: str,
    models: list[str] | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> Any:
    from fa.forecasting.service import run_forecast

    return run_forecast(ws.service(offline), symbol, models=models, refresh=refresh)


# --- leaderboards (cached walk-forward evaluations) -----------------------------------


class LeaderboardFile(BaseModel):
    path: str
    symbol: str
    last_date: date
    models: list[str]
    eval_start: date
    eval_end: date
    folds: int
    modified: datetime


def list_leaderboards(ws: Workspace) -> list[LeaderboardFile]:
    from fa.forecasting.service import CachedEval

    out = []
    for p in sorted(ws.leaderboards_dir.glob("*.json")):
        try:
            ev = CachedEval.model_validate_json(p.read_text())
        except ValueError:
            continue
        stem, _, _key = p.stem.rpartition("_")
        sym, _, last = stem.rpartition("_")
        out.append(
            LeaderboardFile(
                path=str(p),
                symbol=sym.replace("_", "="),
                last_date=date.fromisoformat(last),
                models=sorted({r.model for r in ev.leaderboard}),
                eval_start=ev.eval_start,
                eval_end=ev.eval_end,
                folds=ev.folds,
                modified=datetime.fromtimestamp(p.stat().st_mtime),
            )
        )
    return sorted(out, key=lambda f: f.modified, reverse=True)


def load_leaderboard(path: str | Path) -> pd.DataFrame:
    from fa.forecasting.service import CachedEval

    ev = CachedEval.model_validate_json(Path(path).read_text())
    return pd.DataFrame([r.model_dump() for r in ev.leaderboard])


# --- agent reports --------------------------------------------------------------------


class ReportFile(BaseModel):
    path: str
    symbol: str
    as_of: date
    rating: str
    conviction: float
    cost_usd: float
    generated_at: datetime
    lookahead_warning: bool


def list_reports(ws: Workspace) -> list[ReportFile]:
    out = []
    for p in ws.reports_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            out.append(
                ReportFile(
                    path=str(p),
                    symbol=d["symbol"],
                    as_of=d["as_of"],
                    rating=d["rating"],
                    conviction=d["conviction"],
                    cost_usd=d["cost"]["usd"],
                    generated_at=d["generated_at"],
                    lookahead_warning=bool(d.get("lookahead_warning")),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue  # not a report
    return sorted(out, key=lambda r: r.generated_at, reverse=True)


def load_report(path: str | Path) -> Any:
    from fa.reports.schema import Report

    return Report.model_validate_json(Path(path).read_text())


def run_report(
    ws: Workspace,
    symbol: str,
    as_of: date | None = None,
    offline: bool = False,
    debate_rounds: int | None = None,
    budget_usd: float | None = None,
    llm: Any = None,
) -> tuple[Any, dict[str, Path]]:
    """Run the agent pipeline and save the report. Raises PipelineAborted on budget."""
    from fa.agents.base import AnthropicLLM
    from fa.orchestration.pipeline import run_report as _run
    from fa.reports.render import save

    cfg = ws.cfg
    if budget_usd is not None:
        b = cfg.models.budget.model_copy(update={"max_usd_per_run": budget_usd})
        cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"budget": b})})
    rep = _run(
        cfg,
        ws.service(offline),
        symbol,
        llm or AnthropicLLM(),
        as_of=as_of,
        runs_dir=ws.runs_dir,
        debate_rounds=debate_rounds,
    )
    return rep, save(rep, ws.reports_dir)


# --- backtests ------------------------------------------------------------------------


class BacktestFile(BaseModel):
    path: str
    symbol: str
    strategy: str
    start: str
    end: str
    cagr: float | None
    sharpe: float | None
    benchmark_cagr: float | None
    edge_p_value: float | None
    generated_at: datetime


def run_backtest(
    ws: Workspace,
    symbol: str,
    strategy: str,
    offline: bool = False,
    confirm: bool = False,
    llm: Any = None,
) -> Any:
    """Paper backtest vs buy-and-hold; saved under the backtests dir. The agent strategy
    raises BacktestCostWarning when the estimate exceeds the budget and not `confirm`."""
    from fa.backtest.service import run_strategy

    kw: dict[str, Any] = {}
    if strategy == "agent":
        kw = {"confirm": confirm, "llm": llm, "runs_dir": ws.runs_dir}
    return run_strategy(ws.service(offline), symbol, strategy, **kw)  # type: ignore[arg-type]


def list_backtests(ws: Workspace) -> list[BacktestFile]:
    out = []
    for p in ws.backtests_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            s, b = d["strategy_metrics"], d["benchmark_metrics"]
            out.append(
                BacktestFile(
                    path=str(p),
                    symbol=d["symbol"],
                    strategy=d["strategy"],
                    start=s["start"],
                    end=s["end"],
                    cagr=s["cagr"],
                    sharpe=s["sharpe"],
                    benchmark_cagr=b["cagr"],
                    edge_p_value=d.get("edge_p_value"),
                    generated_at=d["generated_at"],
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda r: r.generated_at, reverse=True)


def load_backtest(path: str | Path) -> tuple[Any, pd.DataFrame | None]:
    """The report and its daily curves (strategy/benchmark equity, position, cost)."""
    from fa.backtest.service import BacktestReport

    p = Path(path)
    rep = BacktestReport.model_validate_json(p.read_text())
    csv = p.with_suffix(".csv")
    curves = pd.read_csv(csv, index_col=0, parse_dates=True) if csv.exists() else None
    return rep, curves


# --- run history (scratchpads) --------------------------------------------------------


class RunSummary(BaseModel):
    path: str
    run_id: str
    started: datetime | None
    kind: str  # report | question | backtest decision | other
    symbol: str | None
    events: int
    llm_calls: int
    tool_calls: int
    errors: int
    cost_usd: float
    tokens: int
    rating: str | None


def _summarize_run(p: Path) -> RunSummary:
    from fa.orchestration.scratchpad import read_events

    ev = read_events(p)
    by = {e["event"] for e in ev}
    start = next((e for e in ev if e["event"] == "run_start"), None)
    rep = next((e for e in ev if e["event"] == "report"), None)
    llm = [e for e in ev if e["event"] == "llm_call"]
    kind = (
        ("question" if start.get("kind") == "question" else "report")
        if start
        else "backtest decision"
        if "backtest_decision" in by
        else "backtest decision (incomplete)"
        if llm
        else "other"
    )
    decision = next((e for e in ev if e["event"] == "backtest_decision"), None)
    usage = [e.get("usage") or {} for e in llm]
    return RunSummary(
        path=str(p),
        run_id=p.stem.rpartition("_")[2],
        started=ev[0]["ts"] if ev else None,
        kind=kind,
        symbol=(start or {}).get("symbol"),
        events=len(ev),
        llm_calls=len(llm),
        tool_calls=sum(e["event"] == "tool_call" for e in ev),
        errors=sum(e["event"] == "error" for e in ev),
        cost_usd=round(sum(e.get("cost_usd") or 0.0 for e in llm), 6),
        tokens=sum(
            int(u.get("input_tokens", 0) or 0) + int(u.get("output_tokens", 0) or 0) for u in usage
        ),
        rating=(rep or {}).get("rating") or (decision or {}).get("rating"),
    )


def list_runs(ws: Workspace, limit: int = 200) -> list[RunSummary]:
    paths = sorted(ws.runs_dir.glob("*.jsonl"), reverse=True)[:limit]
    out = []
    for p in paths:
        try:
            out.append(_summarize_run(p))
        except (ValueError, KeyError, OSError):
            continue
    return out


def run_events(path: str | Path) -> list[dict[str, Any]]:
    from fa.orchestration.scratchpad import read_events

    return read_events(Path(path))


def events_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per event with the commonly useful columns (payloads stay in `events`)."""
    rows = []
    for e in events:
        rows.append(
            {
                "seq": e.get("seq"),
                "time": str(e.get("ts", ""))[11:23],  # HH:MM:SS.mmm (UTC)
                "event": e.get("event"),
                "agent": e.get("agent"),
                "tool": e.get("tool"),
                "result_id": e.get("result_id"),
                "is_error": e.get("is_error"),
                "model": e.get("model"),
                "cost_usd": e.get("cost_usd"),
                "detail": e.get("error") or e.get("kind") or e.get("stop_reason"),
            }
        )
    return pd.DataFrame(rows)
