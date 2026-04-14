# SwingScan developer targets.
#
# Run from the repo root. Assumes Python 3.11 is available as `py -3.11` on
# Windows, `python3.11` on Linux/macOS. The `VENV` and `PY` variables below
# adapt to whichever shell Make picks up.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Virtualenv location. Override on the command line if you want a shared env:
#   make install VENV=../.venvs/swingscan
VENV ?= .venv

# Pick a platform-appropriate venv layout. Windows venvs put binaries in
# Scripts/, POSIX in bin/.
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
    VENV_PY  := $(VENV_BIN)/python.exe
    BOOTSTRAP_PY := py -3.11
else
    VENV_BIN := $(VENV)/bin
    VENV_PY  := $(VENV_BIN)/python
    BOOTSTRAP_PY := python3.11
endif

.PHONY: help install lint format typecheck test test-cov demo clean version

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

$(VENV_PY):
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel

install: $(VENV_PY)  ## Create venv and install package with dev extras.
	$(VENV_PY) -m pip install -e ".[dev]"
	@echo ""
	@echo "Stage 0 install complete. Stage 1+ will also need the 'pipeline' extra:"
	@echo "  $(VENV_PY) -m pip install -e \".[dev,pipeline]\""

lint: $(VENV_PY)  ## Run ruff lint checks.
	$(VENV_PY) -m ruff check src tests scripts

format: $(VENV_PY)  ## Auto-format with ruff.
	$(VENV_PY) -m ruff format src tests scripts
	$(VENV_PY) -m ruff check --fix src tests scripts

typecheck: $(VENV_PY)  ## Run mypy strict on the swingscan package.
	$(VENV_PY) -m mypy

test: $(VENV_PY)  ## Run pytest.
	$(VENV_PY) -m pytest

test-cov: $(VENV_PY)  ## Run pytest with a coverage HTML report.
	$(VENV_PY) -m pytest --cov-report=html

demo: $(VENV_PY)  ## Launch the Gradio demo (Stage 8+). Placeholder until implemented.
	@echo "Demo target is a placeholder until Stage 8 lands."
	@echo "When implemented: $(VENV_PY) scripts/demo_local.py"
	@exit 1

version: $(VENV_PY)  ## Print the installed swingscan version.
	$(VENV_PY) -m swingscan.cli version

clean:  ## Remove build, cache, and coverage artifacts. Keeps .venv.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find src tests scripts -type d -name __pycache__ -exec rm -rf {} +
