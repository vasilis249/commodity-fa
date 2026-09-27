"""Schema regression: saved outputs (reports, forecasts, backtests, snapshots, eval
results) must not change shape silently. If a change is intentional, bump
SCHEMA_VERSION in fa/reports/schema.py (for the report) and run `make schemas`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fa.reports.render import to_html, to_markdown
from fa.reports.schema import SCHEMA_VERSION, Report
from tests.conftest import FIXTURES

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from update_schemas import OUT, models

HINT = "schema changed: if intentional, bump SCHEMA_VERSION (report) and run `make schemas`"


@pytest.mark.parametrize("name", sorted(models()))
def test_schema_matches_snapshot(name: str) -> None:
    stored = json.loads((OUT / f"{name}.json").read_text())
    current = json.loads(json.dumps(models()[name].model_json_schema(), sort_keys=True))
    assert current == stored, f"{name}: {HINT}"


def test_report_snapshot_carries_the_version() -> None:
    stored = json.loads((OUT / "report.json").read_text())
    assert stored["properties"]["schema_version"]["default"] == SCHEMA_VERSION


def test_saved_v1_report_still_loads_and_renders() -> None:
    """Reports written by earlier versions must stay readable (dashboard, re-render)."""
    rep = Report.model_validate_json((FIXTURES / "report_v1.json").read_text())
    assert rep.schema_version == "1"
    md, html = to_markdown(rep), to_html(rep)
    assert rep.rating in md and rep.disclaimer.split(".")[0] in html
    for section in ("Forecast", "Bull", "Bear", "Risks"):
        assert section.lower() in md.lower()
