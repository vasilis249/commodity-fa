"""Event-driven daily paper backtester.

Timing (no look-ahead by construction):
- a signal decides a target weight at the close of day t, using data up to t;
- the engine trades to that target at the OPEN of day t+1 and pays costs there;
- day t+1's P&L = old weight x overnight move (close t -> open t+1)
                 + new weight x intraday move (open t+1 -> close t+1).

Positions are fractions of equity (+1 = fully long), held at constant weight between
decisions (drift trades are ignored). `pnl_bars` turns raw futures bars into bars whose legs
are only what a held position really earned: a contract splice (in the overnight gap of a
roll-masked session) earns nothing, and non-positive or missing prices earn nothing.
An optional per-roll cost is charged on roll sessions. Paper only: nothing here can place
an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fa.config import BacktestCosts


@dataclass(frozen=True)
class CostModel:
    per_side: float  # fraction of traded notional (commission/fees)
    slippage: float  # fraction of traded notional
    per_roll: float  # fraction of held notional, charged on roll sessions

    @classmethod
    def from_config(cls, c: BacktestCosts) -> CostModel:
        return cls(c.cost_bps_per_side / 1e4, c.slippage_bps_per_side / 1e4, c.roll_cost_bps / 1e4)

    def trade_cost(self, traded: float) -> float:
        return abs(traded) * (self.per_side + self.slippage)


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame  # per day: target, position, trade, cost, ret, equity
    name: str

    @property
    def returns(self) -> pd.Series:
        return self.frame["ret"]

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]


def pnl_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Synthetic open/close whose overnight and intraday ratios are the returns a held
    front-month position earned, plus `zeroed` ("overnight" | "both" | "") per session.

    - valid sessions: raw overnight (open / previous close) and intraday (close / open)
      ratios, identical to the roll-adjusted ones;
    - roll-masked sessions: the overnight gap may hold the vendor's contract splice, so it
      earns zero; the intraday leg (one contract, open to close) is credited. Which
      sessions are masked is used for accounting only, never by a signal;
    - non-positive or missing prices: both legs earn zero (no return is defined).
    Measured on the cached universe, roll-session overnight gaps have a p99 up to 8x the
    normal one, while roll-session intraday moves look like any other session's.
    """
    o = frame["open"].astype(float)
    c = frame["close"].astype(float)
    reason = frame["mask_reason"].fillna("missing")
    pc = c.shift(1)
    ok_c, ok_o, ok_pc = c > 0, o > 0, pc > 0  # NaN compares False
    valid, roll = reason == "", reason == "roll"
    on = np.where(valid & ok_o & ok_pc, o / pc, 1.0)
    intra = np.where(
        (valid | roll) & ok_c & ok_o, c / o, np.where(valid & ok_c & ok_pc & ~ok_o, c / pc, 1.0)
    )
    on[0] = intra[0] = 1.0
    level = np.cumprod(on * intra)
    prev = np.concatenate([[1.0], level[:-1]])
    zeroed = np.where(roll, "overnight", np.where(valid, "", "both"))
    zeroed[0] = ""
    return pd.DataFrame(
        {"open": prev * on, "close": level, "zeroed": zeroed, "mask_reason": reason},
        index=frame.index,
    )


def run_backtest(
    bars: pd.DataFrame,
    targets: pd.Series,
    costs: CostModel,
    name: str,
    roll_days: pd.Series | None = None,
) -> BacktestResult:
    """`bars`: open/close (roll-adjusted); `targets`: weight decided at each close
    (NaN = keep the previous target)."""
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    tgt = targets.reindex(bars.index).to_numpy(dtype=float)
    rolls = (
        roll_days.reindex(bars.index, fill_value=False).to_numpy(dtype=bool)
        if roll_days is not None
        else np.zeros(len(bars), dtype=bool)
    )
    n = len(bars)
    pos_arr, trade_arr, cost_arr, ret_arr, eq_arr, tgt_arr = (np.zeros(n) for _ in range(6))
    equity, pos, pending = 1.0, 0.0, 0.0
    for i in range(n):
        start = equity
        if i > 0:
            o = opens[i] if np.isfinite(opens[i]) and opens[i] > 0 else closes[i - 1]
            equity *= 1 + pos * (o / closes[i - 1] - 1)  # overnight on the old weight
            traded = pending - pos
            cost = costs.trade_cost(traded)
            if rolls[i] and pending != 0:
                cost += abs(pending) * costs.per_roll
            equity *= 1 - cost
            pos = pending
            equity *= 1 + pos * (closes[i] / o - 1)  # intraday on the new weight
            trade_arr[i], cost_arr[i] = traded, cost
        if np.isfinite(tgt[i]):
            pending = float(np.clip(tgt[i], -1.0, 1.0))  # decided at this close
        pos_arr[i], tgt_arr[i] = pos, pending
        ret_arr[i] = equity / start - 1
        eq_arr[i] = equity
    frame = pd.DataFrame(
        {
            "target": tgt_arr,
            "position": pos_arr,
            "trade": trade_arr,
            "cost": cost_arr,
            "ret": ret_arr,
            "equity": eq_arr,
        },
        index=bars.index,
    )
    return BacktestResult(frame, name)


def buy_and_hold(
    bars: pd.DataFrame, costs: CostModel, roll_days: pd.Series | None = None
) -> BacktestResult:
    """Benchmark: decide +1 at the first close, enter at the next open, hold (and roll)."""
    targets = pd.Series(np.nan, index=bars.index)
    targets.iloc[0] = 1.0
    return run_backtest(bars, targets, costs, "buy_and_hold", roll_days)
