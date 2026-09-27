from __future__ import annotations

import shutil
from pathlib import Path

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
