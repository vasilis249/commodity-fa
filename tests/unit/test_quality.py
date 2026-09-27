from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from fa.config import AppConfig, Instrument, RollRule
from fa.data.quality import Severity, check_prices
from fa.data.rolls import roll_adjust
from tests.conftest import load_yahoo_fixture

ETF = Instrument(
    symbol="TEST",
    name="t",
    asset_class="etf",
    sector="x",
    exchange="NYSE Arca",
    currency="USD",
    unit="USD/share",
    settle_time=time(16),
    timezone="America/New_York",
    roll_rule=RollRule.NONE,
)


def _frame(closes: list[float], start: str = "2026-03-02", volume: float = 1000.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": volume, "splits": 0.0}
    )


def _returns(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["close"] / frame["close"].shift(1))


def test_clean_series_has_no_warnings(cfg: AppConfig) -> None:
    rng = np.random.default_rng(0)
    f = _frame(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))))
    rep = check_prices(f, ETF, cfg.data.quality, returns=_returns(f), today=date(2026, 5, 25))
    assert rep.ok
    assert not [i for i in rep.issues if i.severity is not Severity.INFO]


def test_detects_gap_repeats_zero_volume_and_bad_ohlc(cfg: AppConfig) -> None:
    f = _frame([100.0 + i for i in range(40)])
    f = f.drop(f.index[10:15])  # 5 missing sessions
    f.iloc[20:25, f.columns.get_loc("close")] = 50.0  # repeated closes
    f.iloc[30, f.columns.get_loc("volume")] = 0
    f.iloc[31, f.columns.get_loc("high")] = 1.0  # high below close
    rep = check_prices(f, ETF, cfg.data.quality, today=f.index[-1].date())
    for check in ["gap", "repeated_close", "zero_volume", "ohlc_inconsistent"]:
        assert rep.by_check(check) is not None, check


def test_stale_data(cfg: AppConfig) -> None:
    f = _frame([100.0] * 10, start="2026-09-01")
    rep = check_prices(f, ETF, cfg.data.quality, today=date(2026, 9, 25))
    stale = rep.by_check("stale_data")
    assert stale is not None and stale.count == 8  # 15..24 Sep: last bar is 14 Sep


def test_unadjusted_split_is_suspected(cfg: AppConfig) -> None:
    closes = [40.0] * 20 + [10.0] * 20  # looks like a 1:4 split nobody adjusted
    f = _frame(closes)
    rep = check_prices(f, ETF, cfg.data.quality, returns=_returns(f), today=f.index[-1].date())
    issue = rep.by_check("split_suspect")
    assert issue is not None and issue.first == f.index[20].date()


def test_empty_frame_is_an_error(cfg: AppConfig) -> None:
    rep = check_prices(_frame([]).iloc[:0], ETF, cfg.data.quality)
    assert not rep.ok


def test_real_wti_2020_flags_negative_price_and_rolls(cfg: AppConfig) -> None:
    inst = cfg.data.instrument("CL=F")
    assert inst is not None
    prices = load_yahoo_fixture("CL=F", "2020")
    adjusted, events = roll_adjust(prices, inst.roll_rule, "cme", cfg.data.rolls)
    rep = check_prices(
        prices, inst, cfg.data.quality, events, adjusted["ret"], today=date(2020, 6, 1)
    )
    neg = rep.by_check("nonpositive_price")
    assert neg is not None and neg.first == date(2020, 4, 20)
    assert rep.by_check("rolls") is not None
