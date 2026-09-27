"""Forecasting dataset: causal features, horizon labels and truncated views.

Rules (see CLAUDE.md, leakage section):
- Features use returns masked by the *calendar* roll windows (known in advance), never
  the volume-detected mask, which has hindsight.
- Features are scale-free: returns, ratios and indicators on the index
  L_t = exp(cumsum r) anchored at the *first* bar, so no value depends on later data.
- Fundamentals enter through `pit_join` on `available_at` (release time).
- Labels use the final (volume-detected) mask: they are outcomes, never inputs.
- `Dataset.until(pos)` gives the world as seen at row `pos`. Its labels exist only
  where t + h <= pos, which purges overlapping labels by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, time

import numpy as np
import pandas as pd

from fa.analytics import indicators as ind
from fa.analytics.supply_demand import cot_positioning, inventory_vs_seasonal
from fa.config import Instrument, RollRule, RollSettings
from fa.data.calendars import calendar_for_exchange
from fa.data.http import DataUnavailable
from fa.data.pit import pit_join
from fa.data.rolls import calendar_roll_windows, expiries
from fa.data.service import DataService

MIN_FRACTION = 0.6  # share of valid (unmasked) days a rolling window needs


def _rolling_sum(r: pd.Series, n: int) -> pd.Series:
    return r.rolling(n, min_periods=int(np.ceil(n * MIN_FRACTION))).sum()


def feature_returns(
    prices: pd.DataFrame, rule: RollRule, calendar: str, rolls: RollSettings
) -> tuple[pd.Series, pd.Series]:
    """(calendar-masked log returns, roll-window flag). Causal."""
    raw = ind.log_returns(prices["close"].astype(float))
    in_window = calendar_roll_windows(prices.index, rule, calendar, rolls)
    return raw.where(~in_window).rename("r"), in_window


def price_features(
    prices: pd.DataFrame, rule: RollRule, calendar: str, rolls: RollSettings
) -> pd.DataFrame:
    """Scale-free technical and calendar features from OHLCV. Row t uses rows <= t."""
    r, in_window = feature_returns(prices, rule, calendar, rolls)
    level = np.exp(r.fillna(0.0).cumsum())  # anchored at the first bar: causal
    f = pd.DataFrame(index=prices.index)
    f["r1"] = r
    for n in (5, 10, 20, 60, 120):
        f[f"cum_{n}"] = _rolling_sum(r, n)
    for n in (5, 20, 60):
        f[f"vol_{n}"] = ind.realized_vol(r, n, min_fraction=MIN_FRACTION)
    f["vol_ratio_5_60"] = f["vol_5"] / f["vol_60"]
    f["vol_ratio_20_60"] = f["vol_20"] / f["vol_60"]
    f["rsi14"] = ind.rsi(level, 14)
    f["macd_hist_pct"] = ind.macd(level)["macd_hist"] / level
    f["bb_pct_b"] = ind.bollinger(level)["bb_pct_b"]
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = level / ind.sma(level, n) - 1
    f["dd_252"] = level / level.rolling(252, min_periods=60).max() - 1
    # High, low and volume of a daily bar may include trading after the settlement that
    # defines the close (and the decision time), so these features use the previous bar.
    close = prices["close"].astype(float)
    day_range = ((prices["high"] - prices["low"]) / close).where((close > 0) & ~in_window)
    f["range_pct_14"] = day_range.rolling(14, min_periods=8).mean().shift(1)
    if "volume" in prices:
        logv = np.log(prices["volume"].astype(float).where(lambda v: v > 0)).where(~in_window)
        mean, sd = logv.rolling(20, min_periods=12).mean(), logv.rolling(20, min_periods=12).std()
        f["volume_z20"] = ((logv - mean) / sd).shift(1)
    f["dow"] = prices.index.dayofweek
    f["month_sin"] = np.sin(2 * np.pi * prices.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * prices.index.month / 12)
    f["in_roll_window"] = in_window.astype(float)
    f["sessions_to_expiry"] = _sessions_to_expiry(prices.index, rule, calendar)
    return f.replace([np.inf, -np.inf], np.nan)


def _sessions_to_expiry(index: pd.DatetimeIndex, rule: RollRule, calendar: str) -> pd.Series:
    if rule is RollRule.NONE or len(index) == 0:
        return pd.Series(np.nan, index=index)
    exp = expiries(rule, index[0].date(), (index[-1] + pd.Timedelta(days=120)).date(), calendar)
    exp_dates = exp.index.to_numpy()
    nxt = exp_dates[np.searchsorted(exp_dates, index.to_numpy(), side="left")]
    # business days as a calendar-free proxy (holidays shift this by <= 1)
    return pd.Series(
        np.busday_count(index.to_numpy().astype("M8[D]"), nxt.astype("M8[D]")),
        index=index,
        dtype=float,
    )


def fundamental_features(
    sessions: pd.DatetimeIndex,
    eia: dict[str, pd.DataFrame],
    cot: pd.DataFrame | None,
    settle_time: time,
    tz: str,
) -> pd.DataFrame:
    """Weekly fundamentals aligned to sessions by release time (point in time)."""
    out = pd.DataFrame(index=sessions)
    for name, frame in eia.items():
        inv = inventory_vs_seasonal(frame["value"])
        wow_sd = inv["wow"].rolling(52, min_periods=26).std()
        weekly = pd.DataFrame(
            {
                f"{name}_dev_pct": inv["dev_pct"],
                f"{name}_band_pos": inv["band_pos"],
                f"{name}_wow_z": inv["wow_vs_seasonal"] / wow_sd,
                "available_at": frame["available_at"],
            },
            index=frame.index,
        )
        joined = pit_join(sessions, weekly, settle_time, tz)
        for col in weekly.columns.drop("available_at"):
            out[col] = joined[col]
        released = pd.to_datetime(joined["asof_available_at"], utc=True).dt.tz_localize(None)
        out[f"{name}_age_days"] = (pd.Series(joined.index, index=joined.index) - released).dt.days
    if cot is not None and not cot.empty:
        pos = cot_positioning(cot)
        weekly = pd.DataFrame(
            {
                "cot_mm_pctile": pos["mm_net_pctile"],
                "cot_mm_pct_oi": pos["mm_net_pct_oi"],
                "cot_mm_chg_pct_oi": pos["mm_net_change_1w"] / pos["open_interest"],
                "available_at": cot["available_at"],
            },
            index=cot.index,
        )
        joined = pit_join(sessions, weekly, settle_time, tz)
        for col in weekly.columns.drop("available_at"):
            out[col] = joined[col]
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class Dataset:
    symbol: str
    features: pd.DataFrame  # causal, one row per session
    r: pd.Series  # calendar-masked log returns (causal, used by baselines and stats models)
    level: pd.Series  # exp(cumsum r) anchored at the first bar (causal, scale-free)
    close: pd.Series  # raw settle (for price bands at the forecast origin only)
    ret_label: pd.Series | None  # final-mask log returns, for labels only
    horizons: tuple[int, ...]
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.features.index)

    def __len__(self) -> int:
        return len(self.features)

    def until(self, pos: int, with_labels: bool = True) -> Dataset:
        """The dataset as seen at row `pos` (inclusive)."""
        sl = slice(0, pos + 1)
        return replace(
            self,
            features=self.features.iloc[sl],
            r=self.r.iloc[sl],
            level=self.level.iloc[sl],
            close=self.close.iloc[sl],
            ret_label=self.ret_label.iloc[sl]
            if (with_labels and self.ret_label is not None)
            else None,
        )

    def label(self, h: int) -> pd.Series:
        """Sum of log returns over (t, t+h]; NaN where t+h is beyond this view.

        h = 1 needs the next return to be valid; longer horizons need half their days
        valid and count masked days (roll splices, bad prices) as zero.
        """
        if self.ret_label is None:
            raise PermissionError("labels are not available in a prediction view")
        r = self.ret_label
        valid = r.notna().astype(float)
        fwd_sum = r.fillna(0.0).rolling(h).sum().shift(-h)
        fwd_valid = valid.rolling(h).sum().shift(-h)
        need = h if h == 1 else np.ceil(h / 2)
        return fwd_sum.where(fwd_valid >= need).rename(f"y_{h}")


def make_dataset(
    symbol: str,
    prices: pd.DataFrame,
    instrument: Instrument,
    rolls: RollSettings,
    horizons: tuple[int, ...],
    eia: dict[str, pd.DataFrame] | None = None,
    cot: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Dataset:
    """Pure constructor from frames (used by tests and by `build_dataset`)."""
    calendar = calendar_for_exchange(instrument.exchange)
    feats = price_features(prices, instrument.roll_rule, calendar, rolls)
    fund = fundamental_features(
        prices.index, eia or {}, cot, instrument.settle_time, instrument.timezone
    )
    features = pd.concat([feats, fund], axis=1)
    r = feats["r1"]
    return Dataset(
        symbol=symbol,
        features=features,
        r=r,
        level=np.exp(r.fillna(0.0).cumsum()).rename("level"),
        close=prices["close"].astype(float),
        ret_label=prices.get("ret"),
        horizons=horizons,
        notes=list(notes or []),
    )


def build_dataset(
    svc: DataService, symbol: str, horizons: tuple[int, ...], as_of: date | None = None
) -> Dataset:
    """Fetch prices and fundamentals through the data service and build the dataset."""
    pdata = svc.prices(symbol, as_of=as_of)
    inst = pdata.instrument
    notes: list[str] = []
    eia: dict[str, pd.DataFrame] = {}
    for name, sid in inst.eia_series.items():
        try:
            eia[name], _ = svc.eia(sid)
        except DataUnavailable as exc:
            notes.append(f"feature group EIA {name} skipped: {exc}")
    cot = None
    if inst.cot_market_code:
        try:
            cot, _ = svc.cot(symbol)
        except DataUnavailable as exc:
            notes.append(f"feature group COT skipped: {exc}")
    return make_dataset(symbol, pdata.frame, inst, svc.cfg.data.rolls, horizons, eia, cot, notes)
