# commodity-fa

A local research tool for **energy-commodity analysis and probabilistic price forecasting**: WTI, Brent, Henry Hub, TTF, heating oil and RBOB, plus ETF proxies. Claude agents form views. Deterministic, tested Python does all the math.

> Research and learning only. Paper trading only: no broker connections and no order execution. **Not investment advice.**

## Status
Phases 0–4 are done: scaffold, data layer, analytics, forecasting and agents. See `CLAUDE.md` for the roadmap and conventions.

## Quick start
```bash
uv sync                  # Python 3.12 + deps
cp .env.example .env     # add your API keys (all free except Anthropic)
uv run fa --help
uv run fa config check   # validate config/*.yaml, show which keys are set
uv run fa universe       # list configured instruments
make check               # lint + typecheck + tests

uv run fa data prices CL=F            # 10y WTI, roll-adjusted returns, quality report (cached)
uv run fa data prices CL=F --offline  # from cache only
uv run fa data eia CL=F               # EIA weekly inventories with release timestamps
uv run fa data cot CL=F               # CFTC managed-money positioning
uv run fa data news CL=F              # recent headlines
uv run fa analyze CL=F --no-llm       # numeric snapshot (technicals, risk, inventories, COT, curve)
uv run fa forecast CL=F               # 1/5/20-day P10/P50/P90 + P(up), with measured skill vs a random walk
uv run fa report CL=F                 # full agent report (needs ANTHROPIC_API_KEY): reports/CL_F_<date>.md/.html/.json
uv run pytest -m network              # live acceptance test (needs internet)
```

## API keys
| Key | Used for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | agents (Phase 4+) | console.anthropic.com |
| `EIA_API_KEY` | inventories, storage, production | https://www.eia.gov/opendata/register.php (free) |
| `FRED_API_KEY` | macro, spot prices | https://fred.stlouisfed.org/docs/api/api_key.html (free) |

CFTC Commitments of Traders data and yfinance prices need no key.

## Forecasts and honesty
`fa forecast` evaluates every model walk-forward and only shows a model's forecast if it beats the naive random walk out of sample. The test is a batch-means Diebold-Mariano test with a Holm correction across all models and horizons. Otherwise the report shows the random-walk band and says **"No measurable edge"**. For energy futures that is the usual result. Optional foundation model: `uv sync --extra foundation` (Chronos-2, CPU).

## Agent reports (`fa report`)
The report pipeline runs five analysts in parallel: supply/demand, technical, news and sentiment, macro, and a forecast interpreter. A bull and a bear researcher then debate for 2 rounds, a risk reviewer suggests a maximum exposure, and a validator and a synthesizer finish the report. Models and effort per role are set in `config/models.yaml`.
- **Every number in the text is checked in code.** Each one must match a value in the tool result the sentence cites (`[T3]`). Unverifiable sentences go back to the agent to fix; any that still fail are removed and listed in the report.
- **The rating is computed in code** from the analysts' stances and the forecast's measured edge, using the weights in `config/agents.yaml`. The synthesizer explains the rating; it cannot change it.
- **Cost:** the default hard budget is $1.00 per run. The run aborts cleanly if it's exceeded. The cost is printed, and each run's full trail (every tool call, result and LLM call) is written to `.runs/<timestamp>_<id>.jsonl`.
