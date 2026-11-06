.PHONY: install lint format types test check audit sample replay

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

types:
	pyright

test:
	pytest -q

# Everything CI runs, in the same order.
check: lint types test
	ruff format --check src tests

sample:
	adjudicator make-sample-data

audit: sample
	adjudicator audit-labels

# Needs recorded cassettes: run once with `adjudicator --replay record run --split dev`.
replay:
	adjudicator --replay replay run --split dev
