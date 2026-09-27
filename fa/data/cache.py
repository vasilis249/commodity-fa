"""Disk cache: one Parquet file + one JSON metadata file per (provider, key).

Layout: `<root>/<provider>/<key>.parquet` and `<key>.meta.json`. Writes are atomic
(tmp file + rename). DuckDB is used only for ad-hoc queries across cached files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel

_SAFE = re.compile(r"[^A-Za-z0-9._=-]")


class CacheMeta(BaseModel):
    provider: str
    key: str
    fetched_at: datetime  # UTC, last network fetch
    coverage_start: date | None = None  # earliest date requested so far (ranged data)
    rows: int
    params: dict[str, str] = {}

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return (now - self.fetched_at).total_seconds() / 3600


@dataclass
class CacheEntry:
    frame: pd.DataFrame
    meta: CacheMeta


def safe_key(key: str) -> str:
    return _SAFE.sub("_", key)


class DiskCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _paths(self, provider: str, key: str) -> tuple[Path, Path]:
        # plain concatenation: keys like "PET.WCESTUS1.W" contain dots
        base = self.root / provider / safe_key(key)
        return Path(f"{base}.parquet"), Path(f"{base}.meta.json")

    def load(self, provider: str, key: str) -> CacheEntry | None:
        data_path, meta_path = self._paths(provider, key)
        if not (data_path.exists() and meta_path.exists()):
            return None
        meta = CacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return CacheEntry(frame=pd.read_parquet(data_path), meta=meta)

    def save(self, frame: pd.DataFrame, meta: CacheMeta) -> None:
        data_path, meta_path = self._paths(meta.provider, meta.key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(data_path, frame.to_parquet())
        _atomic_write_bytes(meta_path, meta.model_dump_json(indent=2).encode())

    def entries(self) -> list[CacheMeta]:
        metas = sorted(self.root.glob("*/*.meta.json")) if self.root.exists() else []
        return [CacheMeta.model_validate_json(p.read_text(encoding="utf-8")) for p in metas]

    def query(self, sql: str) -> pd.DataFrame:
        """Run SQL over cached files. Each provider dir is a view, e.g. `SELECT * FROM yahoo`.

        Views expose a `filename` column so rows can be traced to their cache entry.
        """
        con = duckdb.connect()
        try:
            for provider_dir in sorted(p for p in self.root.glob("*") if p.is_dir()):
                files = str(provider_dir / "*.parquet")
                if any(provider_dir.glob("*.parquet")):
                    con.execute(
                        f'CREATE VIEW "{provider_dir.name}" AS SELECT * FROM '
                        f"read_parquet('{files}', filename=true, union_by_name=true)"
                    )
            return con.execute(sql).df()
        finally:
            con.close()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
