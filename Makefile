.PHONY: install lint fmt test
install:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]
lint:
	ruff check .
fmt:
	ruff format .
test:
	pytest
