from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fa.analytics import curve as cv
from fa.analytics import seasonality as seas
from fa.analytics import supply_demand as sd
from fa.data.providers.cftc import parse_rows
from tests.conftest import FIXTURES


def test_monthly_returns_drop_poorly_covered_months() -> None:
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    r = pd.Series(0.001, index=idx)
    r["2025-02-01":"2025-02-20"] = np.nan  # February mostly masked
    m = seas.monthly_returns(r)
    assert [d.month for d in m.index] == [1, 3]
    assert m.iloc[0] == pytest.approx(0.001 * len(r["2025-01"]))


def test_month_stats_use_only_completed_past_months() -> None:
    idx = pd.to_datetime([f"{y}-09-30" for y in range(2018, 2026)])
    monthly = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.05, 0.9], index=idx)
    st = seas.month_stats(monthly, 9, pd.Timestamp("2025-09-15"))  # Sep 2025 not finished
    assert st.years == 7 and st.last_year == 2024
    assert st.mean == pytest.approx(np.mean([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.05]))
    assert st.hit_rate == pytest.approx(5 / 7)


def _weekly(values_by_year: dict[int, float]) -> pd.Series:
    out = {}
    for year, v in values_by_year.items():
        for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-FRI"):
            out[d] = v + d.isocalendar().week  # same seasonal shape each year
    return pd.Series(out)


def test_seasonal_band_uses_only_prior_years() -> None:
    s = _weekly({2019: 0, 2020: 10, 2021: 20, 2022: 30, 2023: 40, 2024: 50, 2025: 60})
    band = seas.seasonal_band(s, years=5)
    row = band.loc[s["2025"].index[10]]
    week = s["2025"].index[10].isocalendar().week
    assert row["n_years"] == 5
    assert row["avg"] == pytest.approx(np.mean([10, 20, 30, 40, 50]) + week)
    # leak check: changing 2025 values does not move 2025's own band
    s2 = s.copy()
    s2["2025"] += 1000
    pd.testing.assert_series_equal(seas.seasonal_band(s2)["avg"]["2025"], band["avg"]["2025"])


def test_week_53_is_folded() -> None:
    idx = pd.DatetimeIndex(["2020-12-31", "2021-01-01"])  # ISO week 53 of 2020
    assert list(seas.week_of_year(idx)) == [52, 52]


def test_inventory_vs_seasonal() -> None:
    s = _weekly({2019: 0, 2020: 10, 2021: 20, 2022: 30, 2023: 40, 2024: 50, 2025: 100})
    inv = sd.inventory_vs_seasonal(s)
    last = inv.iloc[-1]
    assert last["dev"] == pytest.approx(100 - 30)  # 5y avg of offsets 0..50 excl. 2019 = 30
    assert last["band_pos"] > 1  # above the 5y max
    assert last["wow_vs_seasonal"] == pytest.approx(0.0)  # same weekly shape every year


def test_cot_positioning_from_recorded_rows() -> None:
    cot = parse_rows(json.loads((FIXTURES / "cftc_disagg_067651.json").read_text()))
    pos = sd.cot_positioning(cot, percentile_weeks=4)
    last = pos.iloc[-1]
    assert last["mm_net"] == 223190 - 121362
    assert last["mm_net_pct_oi"] == pytest.approx((223190 - 121362) / 1841811)
    assert last["commercial_net"] == (607458 + 111924) - (304558 + 584713)


def test_crack_spread_321() -> None:
    idx = pd.bdate_range("2026-01-05", periods=1)
    out = sd.crack_spread_321(pd.Series([70.0], idx), pd.Series([2.0], idx), pd.Series([2.5], idx))
    assert out.iloc[0] == pytest.approx((2 * 84 + 105 - 210) / 3)


def _curve(settles: list[float]) -> pd.DataFrame:
    exp = pd.date_range("2026-10-20", periods=len(settles), freq="MS") + pd.Timedelta(days=19)
    return pd.DataFrame(
        {"contract": [f"C{i}" for i in range(len(settles))], "expiry": exp, "settle": settles}
    )


def test_curve_structures_and_roll_yield() -> None:
    back = cv.curve_metrics(_curve([92.41, 88.71, 86.01]))
    assert back is not None and back.structure == "backwardation"
    assert back.m1_m2_spread == pytest.approx(3.70)
    days = (_curve([1, 2])["expiry"].iloc[1] - _curve([1, 2])["expiry"].iloc[0]).days
    assert back.roll_yield_ann == pytest.approx(np.log(92.41 / 88.71) / (days / 365.25))
    assert cv.curve_metrics(_curve([80, 82, 85])).structure == "contango"  # type: ignore[union-attr]
    assert cv.curve_metrics(_curve([3.2, 3.5, 3.9, 2.8])).structure == "mixed"  # type: ignore[union-attr]
    assert cv.curve_metrics(_curve([50, 50.01])).structure == "flat"  # type: ignore[union-attr]
    assert cv.curve_metrics(_curve([50])) is None
