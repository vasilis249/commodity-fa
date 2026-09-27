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
uv run fa data prices CL=F [--years 10] [--offline] [--refresh] [-v]   # prices + roll masks + quality
uv run fa data eia CL=F | cot CL=F | fred DGS10 | news CL=F
uv run fa forecast CL=F [--models naive,lightgbm] [--folds N] [--refresh] [--json]
uv run fa report CL=F [--as-of D] [--rounds N] [--budget USD]   # agents; needs ANTHROPIC_API_KEY
uv run fa backtest CL=F [-s sma|forecast|agent] [--offline] [--yes] [--json]   # paper backtest vs buy-and-hold
uv run fa data cache                                                     # list cache entries
uv run fa data sql "SELECT count(*) FROM yahoo"                         # DuckDB over .cache/
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

## Data layer (Phase 1)
- Upper layers get data **only** through `fa.data.service.DataService`. It provides `prices()` (a `PriceData` with roll-adjusted frame, roll events and quality report), `eia()`, `cot()`, `fred()` and `news()`.
- Providers (`fa/data/providers/`) only fetch and normalize. Cache (`fa/data/cache.py`): Parquet plus JSON meta per (provider, key). A stale entry refetches only its tail (7-day overlap), and an earlier start fetches only the missing head. `--offline` never touches the network.
- Every non-price series carries `available_at` (UTC). Always align it with `fa.data.pit.pit_join`, which compares against the settlement time (`Instrument.settle_time`/`timezone`).
- Price frames: `ret` is the log return, NaN when masked. `mask_reason` is one of `roll`, `nonpositive_price` or `missing`. `close_adj` is a continuous close anchored at the latest bar.
- **Rolls:** Yahoo's `=F` switch is found as the largest day-over-day volume jump within [expiry−4, expiry+1] sessions. That session and the next are masked; with no jump, the whole window is masked. This masks about 10% of futures returns. Evidence is in the `fa/data/rolls.py` docstring and `tests/fixtures/README.md`.
- **Free-source limits:**
  - Yahoo has no history for expired contracts.
  - EIA stopped publishing futures prices in April 2024.
  - The EIA API serves latest-revision values.
  - Without `EIA_API_KEY`, the EIA provider falls back to `DEMO_KEY` (about 10 requests per hour).
  - FRED needs a key.
  - News RSS is live-only.
- Errors never contain secrets: `fa.data.http.redact` strips `api_key=` and similar from messages.

## Leakage rules (the `leakage-auditor` subagent checks these at the end of phases 2–5)
- **Point-in-time joins at release time, never at observation date.** EIA petroleum data (week to Friday) is released Wednesday 10:30 ET. EIA gas storage comes out Thursday 10:30 ET. COT (as of Tuesday) comes out Friday 15:30 ET. See `config/data.yaml: release_lags`.
- **Continuous futures:** yfinance `=F` series are unadjusted front months. Roll-day returns are fake. Back-adjust using only information known at each date, or mask roll days.
- **The volume-detected roll mask has hindsight.** It picks Yahoo's splice session using volume from up to `window_before` (4) sessions later: the 2026-09 audit found 92 days masked in real time but unmasked in the final mask. Use it for **labels** and `close_adj` only. **Forecasting features must use `fa.data.rolls.calendar_roll_windows`**, which depends only on the exchange calendar and is therefore known in advance.
- **`close_adj` is anchored at the latest close.** Its level at t depends on later roll gaps, so full-sample SMA, MACD, ATR or support levels in price units differ (~2%) from what was known at t. Features must be scale-invariant ratios or returns, or be computed from data truncated at t (the snapshot does the latter).
- **Curve as of a date:** settles must be on or before `as_of`, and contracts contiguous from the front month (`DataService.curve(as_of=...)`).
- **Intraday bar fields:** a bar's high, low and volume may include trading after the settlement (the decision time), so forecasting features lag them one bar.
- **Calendar roll windows** extend to expiry + window_after + mask_after, so they cover every volume-detected splice. `embargo_days` must be ≥ window_before + window_after (the label mask's hindsight); config validation enforces this.
- **Tools honour `as_of`.** News is cut at the as_of settlement, and is live-only beyond 3 days back. Error payloads never echo agent input and can't be cited.
- **Phase 5 anonymization must transform payloads inside `ToolRegistry.call`**, before `compact` and the scratchpad, so the claim checker validates exactly what the model saw. It must strip dates, contract codes, units and benchmark names, and rebase prices. Tool descriptions and per-agent tool sets also reveal the commodity. The Phase 4 audit listed all of these.
- **Model selection is itself a fit.** Never report the leaderboard winner's skill as out of sample without the `auto_select` comparison.
- **Publisher disruptions:** `config/data.yaml: release_overrides` pushes `available_at` late for government-shutdown periods (CFTC COT 2018–19 and 2025). These dates are conservative upper bounds.
- **Negative prices** (WTI at −37.63 on 2020-04-20) make log returns undefined. Flag them and handle them explicitly, never with a silent `NaN`/`inf`.
- **News is live-only.** Feeds carry recent items only, so news can never be used in backtests or walk-forward features.
- **FRED uses first-release values** (`output_type=4`); EIA values are the latest revision, which is a known and minor revision risk.
- **No full-sample statistics** (scalers, z-scores, percentiles, seasonal bands) inside a walk-forward fold. Fit on the training window only.
- **Purge ≥ the longest horizon, plus an embargo** (enforced by `ForecastingConfig`).
- **A test that deliberately leaks future data must fail** (Phase 3).

## Analytics layer (Phase 2)
- `fa/analytics/indicators.py`: SMA/EMA/Wilder, RSI, MACD, Bollinger, ATR, realized vol, drawdown and rolling percentile. They are causal and match TA-Lib to about 1e-10 (`tests/fixtures/indicator_reference.csv`, regenerated by `scripts/make_indicator_reference.py`; TA-Lib is not a dependency).
- Other modules:
  - `technicals.py`: trend and vol regimes, pivots confirmed k bars late, support/resistance.
  - `risk.py`: Sharpe, Sortino, max drawdown, VaR/CVaR, beta, and `position_size` with hard gates from `config/risk.yaml`.
  - `seasonality.py`: monthly stats and prior-years-only weekly bands.
  - `supply_demand.py`: inventories vs the 5-year band, COT positioning, 3-2-1 crack.
  - `curve.py`: term-structure metrics.
- `snapshot.py` builds the point-in-time `Snapshot` (Pydantic). Agents will cite its numbers in Phase 4. All analytics models inherit `FiniteModel`, so NaN/inf are stored as `None`.
- `fa analyze SYMBOL --no-llm [--as-of D] [--offline] [--json] [--no-curve]`.

## Forecasting (Phase 3)
- `fa/forecasting/dataset.py`: `Dataset` holds causal features, the calendar-masked returns `r` and a scale-free `level` anchored at the first bar. Labels use the final roll mask. `until(pos)` is the world at row `pos`: labels exist only where t + h ≤ pos (purge by construction), and `with_labels=False` raises on `label()`.
- Models (`base.ForecastModel`): `fit(train_view)` and `predict(view, positions)`. Each returns per-horizon frames with `q0.1`, `q0.5`, `q0.9` and `p_up`. The prediction at row p may use rows ≤ p only; `test_models_only_use_rows_up_to_each_origin` checks this.
- The ladder: `naive` (no-change: median 0, P(up) 50%), `drift`, `auto_arima`, `auto_ets`, `lightgbm` (native API, no scikit-learn), and optional `chronos2`. There's also an `ensemble` (inverse past pinball) and `auto_select` (the leader on earlier folds). Both blends only use losses whose labels ended by the fold's training cut.
- Evaluation (`evaluate.py`): walk-forward with an embargo; models are refit every `refit_every` folds. An edge requires positive pinball skill and a batch-means DM test (blocks = origins per fitted model), with a Holm-adjusted p < alpha across **all** model × horizon tests. Direction is tested against the best constant call. The forecast shows the random walk unless a model has an edge, and then discloses the winner's curse and the `auto_select` score.
- The leaderboard is cached in `leaderboards/`, keyed by symbol, last date, config and models.
- **Any new feature** must pass `assert_causal` (`fa/forecasting/leakage.py`), with cut points inside roll windows. Deliberate leaks (shift(-1), full-sample z-score, centered window, bfill, interpolate, period-date join) are tested to be caught.

## Agents (Phase 4)
- Flow (`fa/orchestration/pipeline.py`):
  1. Five analysts run in parallel threads and call tools (`fa/tools/market.py`). Every result gets an id T#, and its exact payload goes to the scratchpad.
  2. A code numeric check runs, then the LLM validator checks meaning and each analyst's stance against its own evidence.
  3. A bull/bear debate (`debate_rounds`), then the risk reviewer.
  4. `decide()` computes the rating in code; the synthesizer explains it, then gets its own checks.
- **Numbers:** `fa/agents/claims.py`. Every digit run in LLM text is a claim, unless it is a whitelisted identifier, a small count, a duration or a year in date context. Each claim must equal a value in a cited, non-error result after rounding to the precision written. Percent claims match fractions ×100, or raw values only in percentage fields. Sentences that still fail after `validator_max_loops` fixes are removed and listed in the report. Statements the validator still flags are removed too. A stance it still flags gets confidence 0.
- **LLM calls** (`fa/agents/base.py`): adaptive thinking, `output_config.effort` plus a JSON-schema `format` (falling back to JSON-by-instruction on a 400), top-level `cache_control`, and an append-only history. There's no temperature and no forced `tool_choice`. `AgentSession.revise()` continues the same conversation.
- **Budget** (`fa/orchestration/budget.py`): each call reserves its estimated cost under a lock (input priced at the cache-write rate) and settles it on charge, so parallel agents can't overspend. `BudgetExceeded` becomes `PipelineAborted` with the scratchpad kept.
- **Prompts** are in `prompts/*.md` with a `version:` header, plus `_common.md`. The version is logged per call. Bump it whenever the text changes.
- **Reports** go to `reports/<SYM>_<date>.{json,md,html}`. A past `as_of` (more than 3 days back) sets `lookahead_warning`: the agents may know what happened next.

## Backtester (Phase 5)
- **Timing** (`fa/backtest/engine.py`): a signal decides a target weight at the close of day t; the engine trades at the open of t+1. Day t+1's P&L is the old weight on the overnight move plus the new weight on the intraday move. Costs (commission + slippage per side) apply to every trade, and the roll cost applies to held notional on the first session after each expiry (`config/risk.yaml: backtest`). Weights are clipped to [-1, 1].
- **P&L bars** (`pnl_bars`): valid sessions use the raw overnight and intraday ratios. On roll-masked sessions the overnight gap (where the vendor's contract splice sits) earns zero, and the intraday leg is credited. Non-positive or missing prices earn zero, and the report warns about them (e.g. CL on 2020-04-20/21). The mask is used for accounting only; **signals never read it**, nor `close_adj`.
- **Signals** (`fa/backtest/signals.py`) read only the causal `Dataset` series (`level`, `r`). SMA crossover and vol targeting are checked with `assert_causal`, and a deliberately peeking signal is tested to be caught. The forecast signal uses walk-forward predictions (every fold) at origins only; its thresholds are fixed in `config/backtest.yaml`, **never tuned on backtest results**.
- **Benchmark:** buy-and-hold over exactly the same window (from the first valid target), with the same costs. `fa/backtest/metrics.py` computes CAGR, vol, Sharpe with a 95% CI (Lo 2002), Sortino, max drawdown, Calmar, hit rate, turnover, exposure, trades and costs paid. A signal invested under 5% of the time gets a note saying so. `edge_pvalue` is a one-sided batch-means test on daily return differences vs buy-and-hold (`significance_block` sessions per block; measured size 2.8% at a nominal 5% on coin flips). The first note says "No measurable edge vs buy-and-hold" unless p < alpha.
- **Agent backtests** (`fa/backtest/agent_signal.py`, `anonymize.py`, `prompts/anon/`): every `rebalance_every` sessions (the last `max_decisions`), the analysts run on **anonymized** tools built from an allow-list: prices rebased to 100 at the first decision, inventories only as ratios and band positions, COT as a share and percentile, and no news, macro, dates, names, units or sources. The forecast only counts when its skill on forecasts **settled before t** (label end + embargo ≤ t) is significant. Every anonymized tool returns the same shape for every asset (nulls with `available: false`, never errors). Inventory slots are labelled by kind (stocks, supply, processing rate). Rebasing uses the last positive close at or before the first decision and fails closed. Notes and params record the model ids, and how many decisions fall before each model's `training_cutoff` (`models.yaml`; unset means all of them are at risk). The rating is computed by `decide()` and mapped to a weight (long-only clips at 0). Cost: `decisions × est_usd_per_decision` must fit `models.budget.max_usd_backtest` unless `--yes`, and a shared `CostTracker` hard-caps the whole run. Residual risk: the price path's shape can still reveal the episode.
- Results are written to `backtests/<SYM>_<strategy>_<end>.{json,csv}` (git-ignored).

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
| 1 ✅ | Providers (yfinance, EIA, CFTC, FRED, RSS), cache, quality checks (gaps, rolls, negative prices, stale data) | 10 years of `CL=F` fetched twice; the second run hits the cache only; tests use fixtures |
| 2 ✅ | Indicators, supply/demand, curve, seasonality, risk metrics | Reference-value tests; `fa analyze CL=F --no-llm` prints a snapshot |
| 3 ✅ | Model ladder, walk-forward, leaderboard | `fa forecast CL=F`/`SPY` shows quantiles and the leaderboard; leakage test passes; honest "no edge" message |
| 4 ✅ | Tool registry, agents, debate, validator, scratchpad, cost tracking | `fa report CL=F` is schema-valid; every number traces to a tool call; cost printed |
| 5 ✅ | Paper backtester, signals, anonymized LLM mode | SMA crossover and forecast signal vs buy-and-hold, with costs |
| 6 | Streamlit dashboard | `uv run streamlit run app/main.py` works on a fresh clone |
| 7 | Eval set (~30 verifiable questions), schema regression, README | Eval score reported; setup takes under 10 minutes |
