.PHONY: install check bot index test

PYTHON ?= .venv/bin/python

install:
	uv sync --extra local-ml --extra ingestion

check:
	$(PYTHON) main.py --check --local

bot:
	$(PYTHON) main.py --bot

index:
	$(PYTHON) -m src.vectordb.vector_store --local

test:
	$(PYTHON) -m pytest -q
