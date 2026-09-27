# commodity-fa

A local research tool for **energy-commodity analysis and probabilistic price forecasting**: WTI, Brent, Henry Hub, TTF, heating oil and RBOB, plus ETF proxies. Claude agents form views. Deterministic, tested Python does all the math.

> Research and learning only. Paper trading only: no broker connections and no order execution. **Not investment advice.**

## Status
Phases 0 (scaffold) and 1 (data layer) are done. See `CLAUDE.md` for the roadmap and conventions.

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
uv run pytest -m network              # live acceptance test (needs internet)
```

## API keys
| Key | Used for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | agents (Phase 4+) | console.anthropic.com |
| `EIA_API_KEY` | inventories, storage, production | https://www.eia.gov/opendata/register.php (free) |
| `FRED_API_KEY` | macro, spot prices | https://fred.stlouisfed.org/docs/api/api_key.html (free) |

CFTC Commitments of Traders data and yfinance prices need no key.
