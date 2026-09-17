.PHONY: install check bot index test

PYTHON ?= .venv/bin/python

install:
	uv sync --extra ingestion

check:
	$(PYTHON) main.py --check

bot:
	$(PYTHON) main.py --bot

index:
	$(PYTHON) -m src.vectordb.vector_store --mode clean

test:
	$(PYTHON) -m pytest -q
