from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fa.config import AppConfig, RollRule
from fa.data.rolls import (
    adjusted_returns,
    contract_code,
    detect_rolls,
    expiries,
    expiry_date,
    roll_adjust,
)
from tests.conftest import load_yahoo_fixture


@pytest.mark.parametrize(
    ("rule", "cal", "year", "month", "expected"),
    [
        # published last trade dates
        (RollRule.NYMEX_CL, "cme", 2024, 12, date(2024, 11, 20)),
        (RollRule.NYMEX_CL, "cme", 2025, 1, date(2024, 12, 19)),  # 25th is Christmas
        (RollRule.NYMEX_CL, "cme", 2020, 5, date(2020, 4, 21)),  # 25th is a Saturday
        (RollRule.NYMEX_CL, "cme", 2026, 10, date(2026, 9, 22)),
        (RollRule.NYMEX_CL, "cme", 2026, 11, date(2026, 10, 20)),  # 25th is a Sunday
        (RollRule.NYMEX_NG, "cme", 2025, 1, date(2024, 12, 27)),
        (RollRule.NYMEX_NG, "cme", 2026, 8, date(2026, 7, 29)),
        (RollRule.NYMEX_NG, "cme", 2026, 10, date(2026, 9, 28)),
        (RollRule.NYMEX_LAST_BD_PRIOR_MONTH, "cme", 2025, 1, date(2024, 12, 31)),
        (RollRule.NYMEX_LAST_BD_PRIOR_MONTH, "cme", 2026, 9, date(2026, 8, 31)),
        (RollRule.ICE_BRENT, "uk", 2025, 3, date(2025, 1, 31)),
        (RollRule.ICE_BRENT, "uk", 2026, 10, date(2026, 8, 28)),  # Aug 31 is a UK holiday
        (RollRule.ICE_BRENT, "uk", 2026, 11, date(2026, 9, 30)),
        (RollRule.ICE_ENDEX_TTF, "uk", 2026, 11, date(2026, 10, 29)),
    ],
)
def test_expiry_dates(rule: RollRule, cal: str, year: int, month: int, expected: date) -> None:
    assert expiry_date(rule, year, month, cal) == expected


def test_contract_code_and_monthly_expiries() -> None:
    assert contract_code("CL", 2026, 12) == "CLZ26"
    exp = expiries(RollRule.NYMEX_CL, date(2026, 1, 1), date(2026, 12, 31), "cme")
    assert len(exp) == 12
    assert exp.index.is_monotonic_increasing
    with pytest.raises(ValueError):
        expiry_date(RollRule.NONE, 2026, 1, "cme")


# Contract switches verified against individual contracts (see tests/fixtures/README.md)
VERIFIED_SWITCHES = [
    ("CL=F", RollRule.NYMEX_CL, "cme", date(2026, 9, 23)),
    ("NG=F", RollRule.NYMEX_NG, "cme", date(2026, 9, 25)),
    ("HO=F", RollRule.NYMEX_LAST_BD_PRIOR_MONTH, "cme", date(2026, 9, 25)),
    ("RB=F", RollRule.NYMEX_LAST_BD_PRIOR_MONTH, "cme", date(2026, 9, 25)),
    ("BZ=F", RollRule.ICE_BRENT, "uk", date(2026, 8, 31)),
    ("BZ=F", RollRule.ICE_BRENT, "uk", date(2026, 9, 25)),  # no volume signal: window mask
]


@pytest.mark.parametrize(("symbol", "rule", "cal", "switch"), VERIFIED_SWITCHES)
def test_verified_switch_days_are_masked(
    cfg: AppConfig, symbol: str, rule: RollRule, cal: str, switch: date
) -> None:
    prices = load_yahoo_fixture(symbol, "2026")
    adjusted, events = roll_adjust(prices, rule, cal, cfg.data.rolls)
    assert adjusted.loc[pd.Timestamp(switch), "mask_reason"] == "roll"
    assert any(switch in ev.masked for ev in events)


def test_masking_is_sparse_when_volume_signal_exists(cfg: AppConfig) -> None:
    prices = load_yahoo_fixture("CL=F", "2026")
    adjusted, events = roll_adjust(prices, RollRule.NYMEX_CL, "cme", cfg.data.rolls)
    assert all(ev.method == "volume" for ev in events)
    assert (adjusted["mask_reason"] == "roll").sum() == 2 * len(events)


def test_negative_wti_is_masked_not_infinite(cfg: AppConfig) -> None:
    prices = load_yahoo_fixture("CL=F", "2020")
    assert prices.loc["2020-04-20", "close"] < 0
    adjusted, _ = roll_adjust(prices, RollRule.NYMEX_CL, "cme", cfg.data.rolls)
    assert adjusted.loc["2020-04-20", "mask_reason"] == "nonpositive_price"
    assert adjusted.loc["2020-04-21", "mask_reason"] == "nonpositive_price"
    assert np.isfinite(adjusted["ret"].dropna()).all()
    assert (adjusted["close_adj"] > 0).all()


def _synthetic(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": volumes or [100.0] * len(c)}
    )


def test_adjusted_close_removes_roll_gap_and_anchors_at_last_close() -> None:
    # +1% per day, a -10% splice on day 5, then +1% per day again
    closes = [100 * 1.01**i for i in range(5)] + [
        100 * 1.01**4 * 0.9 * 1.01**i for i in range(1, 6)
    ]
    frame = _synthetic(closes)
    from fa.data.rolls import RollEvent

    splice = frame.index[5].date()
    ev = RollEvent(
        expiry=splice, delivery="x", switch_session=splice, masked=(splice,), method="volume"
    )
    out = adjusted_returns(frame, [ev])
    assert out["mask_reason"].iloc[5] == "roll"
    assert out["close_adj"].iloc[-1] == pytest.approx(closes[-1])
    ratios = out["close_adj"] / out["close_adj"].shift(1)
    assert ratios.iloc[5] == pytest.approx(1.0)  # the splice is gone
    assert ratios.drop(ratios.index[[0, 5]]).to_numpy() == pytest.approx(1.01)


def test_detect_falls_back_to_window_without_volume_jump(cfg: AppConfig) -> None:
    frame = _synthetic([100.0] * 30)
    exp = pd.Series({frame.index[15]: "2026-02"})
    events = detect_rolls(frame, exp, cfg.data.rolls, "cme")
    assert events[0].method == "calendar_window"
    assert len(events[0].masked) == cfg.data.rolls.window_before + cfg.data.rolls.window_after + 1


def test_detect_picks_largest_volume_jump_in_window(cfg: AppConfig) -> None:
    vols = [100.0] * 30
    vols[13] = 150.0  # small jump: below threshold after normal day
    vols[15] = 400.0  # the switch
    frame = _synthetic([100.0] * 30, vols)
    exp = pd.Series({frame.index[15]: "2026-02"})
    (ev,) = detect_rolls(frame, exp, cfg.data.rolls, "cme")
    assert ev.method == "volume"
    assert ev.switch_session == frame.index[15].date()
    assert ev.masked == (frame.index[15].date(), frame.index[16].date())


def test_etf_has_no_roll_masking(cfg: AppConfig) -> None:
    frame = _synthetic([10.0, 10.1, 10.2, 10.3])
    out, events = roll_adjust(frame, RollRule.NONE, "cme", cfg.data.rolls)
    assert events == []
    assert (out["mask_reason"] == "roll").sum() == 0
