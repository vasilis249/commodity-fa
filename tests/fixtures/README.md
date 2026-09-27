# Test fixtures

Tests read these files. They never call the network (network tests are marked `@pytest.mark.network` and skipped by default).

| File | Source | Recorded | Notes |
|---|---|---|---|
| `yahoo_raw_CL=F_2020.csv` | yfinance `Ticker.history` (raw, tz-aware) | 2026-09-27 | Mar–May 2020: WTI negative settlement on 2020-04-20 |
| `yahoo_raw_{CL,NG,HO,RB,BZ}=F_2026.csv` | yfinance `Ticker.history` | 2026-09-27 | Jun–Sep 2026. The Sep 2026 contract switches were verified against individual contracts (CLX26.NYM etc.) |
| `eia_seriesid_WCESTUS1.json` | EIA API v2 `/seriesid/PET.WCESTUS1.W`, `length=6` | 2026-09-27 | api_key removed from the echoed request |
| `cftc_disagg_067651.json` | CFTC Socrata `72hh-3qpy`, the adapter's own query | 2026-09-27 | WTI, Sep 2026 |
| `fred_observations_DGS10_synthetic.json` | **Synthetic**, following the documented FRED `series/observations` format (`output_type=4`) | — | No FRED key was available when recording. Replace with a real response once one is |
| `rss_google_news.xml` | Google News RSS search, trimmed to 5 items | 2026-09-27 | |
| `rss_eia_todayinenergy.xml` | EIA "Today in Energy" RSS, trimmed to 4 items | 2026-09-27 | |

Verified contract switches in the 2026 files, used as regression tests:

| Front series | Switch visible in close | Volume jump | Expiry (rule) |
|---|---|---|---|
| CL=F | 2026-09-23 (= CLX26) | 2026-09-22 | 2026-09-22 |
| NG=F | 2026-09-25 (= NGX26) | 2026-09-25 | 2026-09-28 |
| HO=F | 2026-09-25 (= HOX26) | 2026-09-25 | 2026-09-30 |
| RB=F | 2026-09-25 (= RBX26) | 2026-09-25 | 2026-09-30 |
| BZ=F | 2026-08-31 (= BZX26), 2026-09-25 (= BZZ26) | 2026-08-28, none | 2026-08-28, 2026-09-30 |
