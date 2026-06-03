# Makefile for gridstate

VENV := .venv
PYTHON := python3
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

HAS_VENV := $(shell test -d $(VENV) && echo 1 || echo 0)

.PHONY: help venv check-venv install install-dev test test-cov lint format type-check clean build setup

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Activate it: source $(VENV_BIN)/activate"

check-venv:
ifeq ($(HAS_VENV),0)
	@echo "Virtual environment not found. Create it: make venv"
	@exit 1
endif

install: check-venv
	$(VENV_PIP) install -e .

install-dev: check-venv
	$(VENV_PIP) install -e ".[dev,test]"
	$(VENV_BIN)/pre-commit install

test: check-venv
	$(VENV_BIN)/pytest tests/ -v

test-cov: check-venv
	$(VENV_BIN)/pytest tests/ -v --cov=gridstate --cov-report=html --cov-report=term

lint: check-venv
	$(VENV_BIN)/ruff check gridstate tests
	$(VENV_BIN)/ruff format --check gridstate tests

format: check-venv
	$(VENV_BIN)/ruff format gridstate tests
	$(VENV_BIN)/ruff check --fix gridstate tests

type-check: check-venv
	$(VENV_BIN)/mypy gridstate

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path './.venv/*' -delete

build: check-venv
	$(VENV_PYTHON) -m build

check: format lint type-check test
	@echo "All checks passed."

setup:
	@$(MAKE) venv
	@$(MAKE) install-dev
	@echo "Environment ready. Activate it: source $(VENV_BIN)/activate"

help:
	@echo "Available targets:"
	@echo "  make setup        Full setup for a new developer"
	@echo "  make venv         Create the virtual environment"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install with dev/test extras and pre-commit hooks"
	@echo "  make test         Run the test suite"
	@echo "  make test-cov     Run tests with a coverage report"
	@echo "  make lint         Style check (ruff)"
	@echo "  make format       Auto-format"
	@echo "  make type-check   Type check (mypy)"
	@echo "  make check        format + lint + type-check + test"
	@echo "  make clean        Remove temporary files"
	@echo "  make build        Build the package"
