"""Regenerate the schema snapshots checked by tests/unit/test_schemas.py.

Run after an intentional change to a saved-output model (and bump SCHEMA_VERSION in
fa/reports/schema.py when the report changes):  uv run python scripts/update_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "schemas"


def models() -> dict[str, type]:
    from fa.agents.schemas import ResearchAnswer
    from fa.analytics.snapshot import Snapshot
    from fa.backtest.service import BacktestReport
    from fa.evals import EvalReport
    from fa.forecasting.service import ForecastReport
    from fa.reports.schema import Report

    return {
        "report": Report,
        "forecast_report": ForecastReport,
        "backtest_report": BacktestReport,
        "snapshot": Snapshot,
        "research_answer": ResearchAnswer,
        "eval_report": EvalReport,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in models().items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
