PYTHON ?= python3
VENV ?= .venv
BIN = $(VENV)/bin

.PHONY: install build test format format-check lint typecheck check
install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
	$(BIN)/python -m pip install -e '.[dev,train]'
build:
	$(BIN)/python -m build --no-isolation
test:
	$(BIN)/python -m pytest
format:
	$(BIN)/ruff format src tests examples
format-check:
	$(BIN)/ruff format --check src tests examples
lint:
	$(BIN)/ruff check src tests examples
typecheck:
	$(BIN)/mypy src/speechturn
check: build test format-check lint typecheck
