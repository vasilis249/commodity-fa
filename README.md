# commodity-fa

A local research tool for **energy-commodity analysis and probabilistic price forecasting**: WTI, Brent, Henry Hub, TTF, heating oil and RBOB, plus ETF proxies. Claude agents form views. Deterministic, tested Python does all the math.

> Research and learning only. Paper trading only: no broker connections and no order execution. **Not investment advice.**

## Status
Phase 0 (scaffold) is done. See `CLAUDE.md` for the roadmap and conventions.

## Quick start
```bash
uv sync                  # Python 3.12 + deps
cp .env.example .env     # add your API keys (all free except Anthropic)
uv run fa --help
uv run fa config check   # validate config/*.yaml, show which keys are set
uv run fa universe       # list configured instruments
make check               # lint + typecheck + tests
```

## API keys
| Key | Used for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | agents (Phase 4+) | console.anthropic.com |
| `EIA_API_KEY` | inventories, storage, production | https://www.eia.gov/opendata/register.php (free) |
| `FRED_API_KEY` | macro, spot prices | https://fred.stlouisfed.org/docs/api/api_key.html (free) |

CFTC Commitments of Traders data and yfinance prices need no key.
