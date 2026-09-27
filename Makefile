.PHONY: install test lint fmt typecheck check app schemas evals

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy fa

check: lint typecheck test

app:
	uv run streamlit run app/main.py

schemas:
	uv run python scripts/update_schemas.py

evals:
	uv run fa eval
