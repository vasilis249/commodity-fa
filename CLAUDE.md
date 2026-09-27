# CLAUDE.md — commodity-fa

A local research tool for **energy-commodity** analysis and probabilistic forecasting. Claude agents write the theses; deterministic, tested Python does all the math.
**Paper only**: no broker APIs, no live trading, no order execution. Every report carries a "not investment advice" note.

## Commands
```bash
uv sync                      # install (Python 3.12, pinned in .python-version)
uv run pytest                # tests (network tests excluded; run them with -m network)
uv run ruff check . && uv run ruff format --check .
uv run mypy fa               # lenient
make check                   # lint + typecheck + tests
uv run fa --help
uv run fa config check       # validate config/*.yaml, show which keys are set
uv run fa universe
```
Optional extras (declared in the phase that needs them): `uv sync --extra foundation` for Chronos-2 on CPU.

## Architecture: layers depend only downward
```
Interface      fa/cli.py (Typer) · app/ (Streamlit)
Orchestration  fa/orchestration/  pipeline, research loop, budget, JSONL scratchpad
Agents         fa/agents/         analysts, bull/bear, risk reviewer, validator, synthesizer
Tools          fa/tools/          Pydantic-schema'd wrappers the agents call
Forecasting    fa/forecasting/    baselines → stats → LightGBM → Chronos-2 → ensemble
Analytics      fa/analytics/      indicators, supply/demand, curve, seasonality, risk (pure)
Data           fa/data/           providers, disk cache, quality checks
Reports        fa/reports/        schema + render (used by orchestration/interface)
Config         fa/config.py       Pydantic models for config/*.yaml + secrets from .env
```
Analytics, forecasting and backtesting must run **without any LLM**, so they stay testable and free to run.

## Design rules (from the brief; these are requirements)
1. **LLMs form views; code does the math and the decisions.** Every number an agent states must come from a logged tool call. Risk limits are hard gates in code. The final rating and conviction are computed in code from `config/agents.yaml: decision_weights`; the synthesizer explains the result and never invents it.
2. **Specialized analysts, then a bull/bear debate** (`debate_rounds`), then the risk reviewer, the validator (≤ `validator_max_loops`) and the synthesizer.
3. **Plan → act → validate → answer** for free-form questions, with a hard step cap (`research_loop_max_steps`).
4. **Log everything** to `.runs/<timestamp>_<id>.jsonl`: tool calls, arguments, a summary of each result, tokens and cost.
5. **Cache every API response** under `.cache/` (Parquet + DuckDB). Reruns work offline once the cache is warm.
6. **No look-ahead, including the LLM's own memory.** Backtests run in anonymized mode: prices rebased to 100, relative periods `t-0, t-1…`, no symbol, name or dates. Even so, a commodity may be recognizable from its behavior, and reports must say so.
7. **Skill must beat the naive baseline out-of-sample.** Otherwise the report says "no measurable edge" plainly.
8. **Config over code.** Universe, roster, rounds, horizons, risk limits, models and prices live in `config/*.yaml`.
9. **Record non-determinism.** Every run logs model, `effort`, prompt version and seeds. Sonnet 5 and Opus 5.5 **reject** `temperature`/`top_p`/`top_k` and `budget_tokens` with a 400, so control depth with `output_config.effort` and adaptive thinking. Opus 5.5 also rejects forced `tool_choice` (`any`/`tool`), so use `auto` plus `strict: true` tools.

## Commodity adaptations of the brief
- **Universe:** `CL=F` WTI, `BZ=F` Brent, `NG=F` Henry Hub, `TTF=F` Dutch TTF, `HO=F` ULSD, `RB=F` RBOB, with USO/UNG/XLE as proxies. Add metals or ags in `config/data.yaml`.
- **Fundamentals** come from **EIA API v2** (legacy series ids through `/v2/seriesid/`) and **CFTC COT** (Socrata, no key), not from SEC EDGAR.
- **The supply/demand analyst replaces the fundamental analyst.** It covers inventories vs the 5-year seasonal band, storage surplus or deficit, crack spreads, term structure and roll yield, and COT percentile.
- **The event calendar** covers EIA weekly releases, OPEC+ meetings, contract expiry, FOMC and hurricane season.

## Leakage rules (the `leakage-auditor` subagent checks these at the end of phases 2–5)
- **Point-in-time joins at release time, never at observation date.** EIA petroleum data (week to Friday) is released Wednesday 10:30 ET. EIA gas storage comes out Thursday 10:30 ET. COT (as of Tuesday) comes out Friday 15:30 ET. See `config/data.yaml: release_lags`.
- **Continuous futures:** yfinance `=F` series are unadjusted front months. Roll-day returns are fake. Back-adjust using only information known at each date, or mask roll days.
- **Negative prices** (WTI at −37.63 on 2020-04-20) make log returns undefined. Flag them and handle them explicitly, never with a silent `NaN`/`inf`.
- **No full-sample statistics** (scalers, z-scores, percentiles, seasonal bands) inside a walk-forward fold. Fit on the training window only.
- **Purge ≥ the longest horizon, plus an embargo** (enforced by `ForecastingConfig`).
- **A test that deliberately leaks future data must fail** (Phase 3).

## Conventions
- Python 3.12, `uv`, `ruff` (line length 100), lenient `mypy`, `pytest`.
- Small, readable modules. Add a dependency only when its phase needs it, and say why.
- Pydantic models for every tool input and output and for the report. Config models use `extra="forbid"`.
- Tests use recorded fixtures in `tests/fixtures/`, never the live network. Live checks are marked `@pytest.mark.network`.
- Indicators are implemented in-house and unit-tested against known reference values. No unmaintained TA libraries.
- Secrets live only in `.env`. They are never written to code, logs, scratchpads or reports (`Secrets.status()` reports set or missing only).
- TimesFM stays off by default because its latest weights have a non-commercial license.
- Default LLM budget is `$1.00` per report (`config/models.yaml`). Abort cleanly when it's exceeded.

## Workflow
Build one phase at a time. At the end of each phase: run the tests, show a demo command, list what's done and what's deferred, commit, push, and **stop for review**.

| Phase | Scope | Acceptance |
|---|---|---|
| 0 ✅ | Scaffold, config, CLI, CLAUDE.md, leakage-auditor | `uv run pytest` green; `uv run fa --help` works |
| 1 | Providers (yfinance, EIA, CFTC, FRED, RSS), cache, quality checks (gaps, rolls, negative prices, stale data) | 10 years of `CL=F` fetched twice; the second run hits the cache only; tests use fixtures |
| 2 | Indicators, supply/demand, curve, seasonality, risk metrics | Reference-value tests; `fa analyze CL=F --no-llm` prints a snapshot |
| 3 | Model ladder, walk-forward, leaderboard | `fa forecast CL=F` shows quantiles and the leaderboard; leakage test passes; honest "no edge" message |
| 4 | Tool registry, agents, debate, validator, scratchpad, cost tracking | `fa report CL=F` is schema-valid; every number traces to a tool call; cost printed |
| 5 | Paper backtester, signals, anonymized LLM mode | SMA crossover and forecast signal vs buy-and-hold, with costs |
| 6 | Streamlit dashboard | `uv run streamlit run app/main.py` works on a fresh clone |
| 7 | Eval set (~30 verifiable questions), schema regression, README | Eval score reported; setup takes under 10 minutes |
