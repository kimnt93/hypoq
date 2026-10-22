# ==============================================================================
# vnresearch — Standalone Quantitative Financial & Macroeconomic Data Lake
# Repository: https://github.com/kimnt93/vnresearch.git
# ==============================================================================

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo $$(pwd)/.venv/bin/python3; elif [ -f ../.venv/bin/python3 ]; then echo $$(cd .. && pwd)/.venv/bin/python3; else which python3; fi)
ANYMD ?= $(shell which anymd || echo "/home/kim/.bun/bin/anymd")

.PHONY: all download verify visualize parse-doc help clean

# Default target downloads and builds all datasets
all: download

help:
	@echo "========================================================================"
	@echo " vnresearch: Autonomous Quantitative Data Lake for Empirical Research"
	@echo "========================================================================"
	@echo " make download         Download, parse with anymd, and build all datasets"
	@echo " make verify           Validate integrity and row counts of all Parquet files"
	@echo " make visualize        Render distribution and overview charts"
	@echo " make parse-doc        Parse document/PDF to Markdown (FILE=<path> [PAGES=1-5])"
	@echo " make clean            Clean temporary download caches and artifacts"
	@echo "========================================================================"

download:
	@echo ">>> Starting Complete Data Ingestion Pipeline via anymd..."
	$(PYTHON) scripts/download_all.py
	@echo ">>> All datasets successfully downloaded and verified!"

verify:
	@echo ">>> Validating Parquet Datasets Integrity..."
	$(PYTHON) scripts/verify_datasets.py

visualize:
	@echo ">>> Rendering Dataset Overview & Distribution Visualizations..."
	$(PYTHON) scripts/render_datasets_overview.py

parse-doc:
	@if [ -z "$(FILE)" ]; then echo "Usage: make parse-doc FILE=<path_or_url> [PAGES=1-5]"; exit 1; fi
	$(ANYMD) $(FILE) $(if $(PAGES),-p $(PAGES),)

clean:
	@echo ">>> Cleaning temporary caches and artifacts..."
	@rm -rf /tmp/widata* /tmp/anymd* .pytest_cache __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "    Done."
