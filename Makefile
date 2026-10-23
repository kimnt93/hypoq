# ==============================================================================
# vnresearch — Standalone Quantitative Financial & Macroeconomic Data Lake
# Repository: https://github.com/kimnt93/vnresearch.git
# ==============================================================================

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo $$(pwd)/.venv/bin/python3; elif [ -f ../.venv/bin/python3 ]; then echo $$(cd .. && pwd)/.venv/bin/python3; else which python3; fi)
ANYMD ?= $(shell which anymd || echo "/home/kim/.bun/bin/anymd")
UV ?= $(shell which uv || echo "/home/kim/.local/bin/uv")

.PHONY: all install download verify parse-doc help clean start stop

# Default target downloads and builds all datasets
all: download

help:
	@echo "========================================================================"
	@echo " hypoq: Autonomous Quantitative Research Lab & Empirical Modeling"
	@echo "========================================================================"
	@echo " make install          Install host research stack (.venv) using uv"
	@echo " make download         Download, parse with anymd, and build all datasets"
	@echo " make verify           Validate integrity and row counts of all Parquet files"
	@echo " make start            Start Next.js web UI display container on Docker"
	@echo " make stop             Stop Next.js web UI display container"
	@echo " make parse-doc        Parse document/PDF to Markdown (FILE=<path> [PAGES=1-5])"
	@echo " make clean            Clean temporary download caches and artifacts"
	@echo "========================================================================"

install:
	@echo ">>> Installing quantitative research stack into .venv using uv..."
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "uv not found, installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	$(UV) sync
	@echo ">>> Research environment successfully configured in .venv!"
	@echo "    Python binary: $(shell pwd)/.venv/bin/python3"
	@$(PYTHON) -c "import polars, statsmodels, sklearn, torch, stable_baselines3; print('>>> Verified Core Libraries: Polars, Statsmodels, Scikit-Learn, PyTorch, Stable-Baselines3')"

download:
	@echo ">>> Starting Complete Data Ingestion Pipeline via anymd..."
	$(PYTHON) scripts/download_all.py
	@echo ">>> All datasets successfully downloaded and verified!"

verify:
	@echo ">>> Validating Parquet Datasets Integrity..."
	$(PYTHON) scripts/verify_datasets.py

parse-doc:
	@if [ -z "$(FILE)" ]; then echo "Usage: make parse-doc FILE=<path_or_url> [PAGES=1-5]"; exit 1; fi
	$(ANYMD) $(FILE) $(if $(PAGES),-p $(PAGES),)

clean:
	@echo ">>> Cleaning temporary caches and artifacts..."
	@rm -rf /tmp/widata* /tmp/anymd* .pytest_cache __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "    Done."

start:
	@echo ">>> Starting hypoq Next.js Full Stack on Docker..."
	docker compose up -d --build --remove-orphans
	@echo ">>> Service live at http://127.0.0.1:$${PORT:-3030}"

stop:
	@echo ">>> Stopping vnresearch stack..."
	docker compose down
	@echo ">>> Service stopped."
