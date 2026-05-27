.PHONY: install install-training install-frontend download train evaluate predict benchmark test dev build clean

PYTHON ?= python
PIP ?= pip

install: install-training install-frontend

install-training:
	cd training && $(PIP) install -r requirements.txt

install-frontend:
	npm install

# ── Training pipeline ──────────────────────────────────
download:
	$(PYTHON) training/download_esc50.py --dest ./data

train:
	cd training && $(PYTHON) train.py

evaluate:
	cd training && $(PYTHON) evaluate.py

predict:
	@if [ -z "$(audio)" ]; then echo "Usage: make predict audio=path/to/sound.wav"; exit 1; fi
	cd training && $(PYTHON) predict.py "$(audio)"

benchmark:
	@if [ -z "$(audio)" ]; then echo "Usage: make benchmark audio=path/to/sound.wav"; exit 1; fi
	cd training && $(PYTHON) benchmark.py --audio "$(audio)" --iterations 200

test:
	cd training && pytest tests/ -v

# ── Browser frontend ───────────────────────────────────
dev:
	npm run dev

build:
	npm run build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist node_modules training/models/*.joblib

help:
	@echo "Common targets:"
	@echo "  make install            Install training + frontend deps"
	@echo "  make download           Download ESC-50 dataset (~600 MB)"
	@echo "  make train              Train the ensemble"
	@echo "  make evaluate           Evaluate on test split"
	@echo "  make predict audio=PATH Run inference on a single file"
	@echo "  make benchmark audio=PATH  Benchmark inference latency"
	@echo "  make test               Run Python unit tests"
	@echo "  make dev                Run browser frontend dev server"
