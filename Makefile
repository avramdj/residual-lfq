.DEFAULT_GOAL := check

.PHONY: format lint typecheck test check clean

format:
	poetry run ruff format .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy .

test:
	poetry run pytest -v

check: format lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 