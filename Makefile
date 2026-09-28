.PHONY: help install install-training install-frontend download test build clean

.DEFAULT_GOAL := help

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

help:
	@echo "Targets:"
	@echo "  make install     Install training and frontend dependencies"
	@echo "  make download    Download the ESC-50 dataset into ./data"
	@echo "  make test        Run the Python and JavaScript tests"
	@echo "  make build       Type-check and build the web app into dist/"
	@echo "  make clean       Remove build output and Python caches"
	@echo ""
	@echo "Training is run by hand for now; see the README."

install: install-training install-frontend

install-training:
	$(PIP) install -r training/requirements.txt

install-frontend:
	npm ci

download:
	$(PYTHON) training/download_esc50.py --dest ./data

test:
	cd training && $(PYTHON) -m pytest tests/ -v
	npm test

build:
	npm run build

clean:
	rm -rf dist training/.pytest_cache
	find . -path ./node_modules -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
