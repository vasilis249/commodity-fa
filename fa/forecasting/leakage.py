"""Leakage detection: a feature function is causal iff truncating its input never
changes its earlier outputs. Used by tests (and available for any new feature)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """A feature at time t changed when data after t was removed."""


def assert_causal(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    cuts: Sequence[int],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Compare build(data) with build(data[:cut]) on rows < cut, for every cut.

    Raises LeakageError naming the first column and date that differ.
    """
    full = build(data)
    for cut in cuts:
        part = build(data.iloc[:cut])
        a = full.iloc[:cut].reindex(columns=part.columns)
        b = part
        close = np.isclose(
            a.to_numpy(float), b.to_numpy(float), rtol=rtol, atol=atol, equal_nan=True
        )
        if not close.all():
            row, col = np.argwhere(~close)[0]
            raise LeakageError(
                f"feature '{b.columns[col]}' at {b.index[row].date()} changes when data after "
                f"{data.index[cut - 1].date()} is removed "
                f"(full={a.iat[row, col]!r}, truncated={b.iat[row, col]!r})"
            )
