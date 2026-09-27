from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from fa.data.cache import CacheMeta, DiskCache


def _meta(provider: str, key: str, rows: int) -> CacheMeta:
    return CacheMeta(
        provider=provider,
        key=key,
        fetched_at=datetime(2026, 9, 27, 12, tzinfo=UTC),
        coverage_start=date(2026, 1, 1),
        rows=rows,
    )


def test_roundtrip_preserves_index_and_tz(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-09-18", "2026-09-25"]), name="date")
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0],
            "available_at": pd.to_datetime(["2026-09-23 14:30", "2026-09-30 14:30"], utc=True),
        },
        index=idx,
    )
    cache.save(frame, _meta("eia", "PET.WCESTUS1.W", 2))
    entry = cache.load("eia", "PET.WCESTUS1.W")
    assert entry is not None
    pd.testing.assert_frame_equal(entry.frame, frame)
    assert entry.meta.coverage_start == date(2026, 1, 1)
    assert not list(tmp_path.rglob("*.tmp"))


def test_dotted_keys_do_not_collide(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    for key, v in [("PET.X.W", 1.0), ("PET.X.D", 2.0)]:
        cache.save(pd.DataFrame({"value": [v]}), _meta("eia", key, 1))
    assert cache.load("eia", "PET.X.W").frame["value"].iloc[0] == 1.0  # type: ignore[union-attr]
    assert cache.load("eia", "PET.X.D").frame["value"].iloc[0] == 2.0  # type: ignore[union-attr]
    assert cache.load("eia", "missing") is None


def test_entries_and_duckdb_query(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-09-24", "2026-09-25"]), name="date")
    cache.save(pd.DataFrame({"close": [94.61, 92.41]}, index=idx), _meta("yahoo", "CL=F", 2))
    cache.save(pd.DataFrame({"close": [3.297, 3.225]}, index=idx), _meta("yahoo", "NG=F", 2))
    assert {m.key for m in cache.entries()} == {"CL=F", "NG=F"}
    out = cache.query("SELECT count(*) AS n, max(close) AS hi FROM yahoo")
    assert out.loc[0, "n"] == 4 and out.loc[0, "hi"] == 94.61
    per_file = cache.query("SELECT filename, count(*) AS n FROM yahoo GROUP BY 1")
    assert len(per_file) == 2
