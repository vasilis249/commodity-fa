"""Leakage tests. `test_deliberate_leaks_are_caught` must FAIL the detector: it proves
the causality check would catch a future-looking feature if one were added."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fa.config import AppConfig, RollRule, load_config
from fa.data.pit import release_timestamp
from fa.data.rolls import calendar_roll_windows
from fa.forecasting.dataset import make_dataset, price_features
from fa.forecasting.leakage import LeakageError, assert_causal
from tests.conftest import FIXTURES

PRICES = pd.read_csv(FIXTURES / "prices_CL=F_2y.csv", index_col=0, parse_dates=True)
# Cut points must include sessions inside a roll window (where features have gaps):
# leaks like bfill/interpolate only show when the cut falls inside a gap.
_WINDOW = calendar_roll_windows(PRICES.index, RollRule.NYMEX_CL, "cme", load_config().data.rolls)
_IN_WINDOW = [int(i) for i in np.flatnonzero(_WINDOW.to_numpy()) if 200 < i < 500][::40][:3]
CUTS = sorted({150, 260, 390, 470, 519, *(i + 1 for i in _IN_WINDOW)})


def _features(cfg: AppConfig):
    return lambda d: price_features(d, RollRule.NYMEX_CL, "cme", cfg.data.rolls)


def test_price_features_are_causal(cfg: AppConfig) -> None:
    assert_causal(_features(cfg), PRICES, CUTS)


@pytest.mark.parametrize(
    "leak",
    [
        lambda f, d: f.assign(next_ret=f["r1"].shift(-1)),  # tomorrow's return
        lambda f, d: f.assign(z=(f["r1"] - f["r1"].mean()) / f["r1"].std()),  # full-sample z-score
        lambda f, d: f.assign(c=d["close"].rolling(5, center=True).mean() / d["close"]),  # centered
        lambda f, d: f.assign(bfilled=f["r1"].bfill()),  # back-fill gaps from the future
        lambda f, d: f.assign(interp=f["r1"].interpolate()),  # interpolate across gaps
    ],
    ids=["shift(-1)", "full_sample_zscore", "centered_window", "bfill", "interpolate"],
)
def test_deliberate_leaks_are_caught(cfg: AppConfig, leak) -> None:
    base = _features(cfg)
    with pytest.raises(LeakageError):
        assert_causal(lambda d: leak(base(d), d), PRICES, CUTS)


def _weekly_fundamentals(cfg: AppConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(3)
    fridays = pd.date_range("2017-01-06", "2026-09-25", freq="W-FRI", name="date")
    eia = pd.DataFrame({"value": 400 + rng.normal(0, 5, len(fridays)).cumsum()}, index=fridays)
    wpsr = cfg.data.releases["eia_petroleum_weekly"]
    eia["available_at"] = [release_timestamp(d.date(), wpsr) for d in fridays]
    tuesdays = pd.date_range("2019-01-01", "2026-09-22", freq="W-TUE", name="date")
    cot = pd.DataFrame(
        {c: rng.integers(100_000, 300_000, len(tuesdays)).astype(float)
         for c in ["prod_long", "prod_short", "swap_long", "swap_short", "mm_long", "mm_short"]},
        index=tuesdays,
    )  # fmt: skip
    cot["open_interest"] = 2e6
    cot["available_at"] = [
        release_timestamp(d.date(), cfg.data.releases["cftc_cot"]) for d in tuesdays
    ]
    return {"stocks": eia}, cot


def test_fundamental_features_are_causal(cfg: AppConfig) -> None:
    """Fundamentals truncated to what existed at the cut give identical features."""
    eia, cot = _weekly_fundamentals(cfg)
    inst = cfg.data.instrument("CL=F")
    assert inst is not None

    def build(d: pd.DataFrame) -> pd.DataFrame:
        end = d.index[-1]
        e = {k: v.loc[:end] for k, v in eia.items()}
        return make_dataset("CL=F", d, inst, cfg.data.rolls, (1,), e, cot.loc[:end]).features

    assert_causal(build, PRICES, CUTS)


def test_prediction_views_have_no_labels(cfg: AppConfig) -> None:
    from tests.forecast_helpers import synthetic_dataset

    ds = synthetic_dataset(cfg, n=300)
    with pytest.raises(PermissionError):
        ds.until(200, with_labels=False).label(1)


@pytest.mark.parametrize("h", [1, 5, 20])
def test_truncated_view_purges_overlapping_labels(cfg: AppConfig, h: int) -> None:
    from tests.forecast_helpers import synthetic_dataset

    ds = synthetic_dataset(cfg, n=300)
    pos = 200
    view = ds.until(pos).label(h)
    full = ds.label(h).iloc[: pos + 1]
    assert view.iloc[pos - h + 1 :].isna().all()  # these would need prices after `pos`
    pd.testing.assert_series_equal(view.iloc[: pos - h + 1], full.iloc[: pos - h + 1])
