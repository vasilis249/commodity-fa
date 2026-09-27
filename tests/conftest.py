from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fa.config import DEFAULT_CONFIG_DIR, AppConfig, load_config


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of config/ for tests that break a file on purpose."""
    dst = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, dst)
    return dst


FIXTURES = Path(__file__).parent / "fixtures"


def load_yahoo_fixture(symbol: str, tag: str) -> pd.DataFrame:
    """Recorded raw yfinance frame -> canonical price frame (via the real normalizer)."""
    from fa.data.providers.yahoo import normalize

    raw = pd.read_csv(FIXTURES / f"yahoo_raw_{symbol}_{tag}.csv", index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
    return normalize(raw)
