# ==============================================================================
# vnresearch — Standalone Quantitative Financial & Macroeconomic Data Lake
# Repository: https://github.com/kimnt93/vnresearch.git
# ==============================================================================

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo $$(pwd)/.venv/bin/python3; elif [ -f ../.venv/bin/python3 ]; then echo $$(cd .. && pwd)/.venv/bin/python3; else which python3; fi)
ANYMD ?= $(shell which anymd || echo "/home/kim/.bun/bin/anymd")
UV ?= $(shell which uv || echo "/home/kim/.local/bin/uv")

.PHONY: all install download verify parse-doc help clean start stop install-pg install-postgres install-sqlite setup-pg setup-sqlite

# Default target downloads and builds all datasets
all: download

help:
	@echo "========================================================================"
	@echo " hypoq: Autonomous Quantitative Research Lab & Empirical Modeling"
	@echo "========================================================================"
	@echo " make install          Install host research stack (.venv) & start default SQLite stack"
	@echo " make download         Download, parse with anymd, and build all datasets"
	@echo " make verify           Validate integrity and row counts of all Parquet files"
	@echo " make start            Start Next.js web UI display container on Docker"
	@echo " make install-pg       Start PostgreSQL with Docker Compose & initialize semantic DB"
	@echo " make install-sqlite   Start SQLite & Next.js web container with Docker Compose"
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
	@echo ">>> Setting up default SQLite database backend and starting web container..."
	$(MAKE) install-sqlite

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

install-pg:
	docker compose up -d postgres
	@echo ">>> Waiting for PostgreSQL container to become ready..."
	@until docker compose exec -T postgres pg_isready -U hypoq -d hypoq >/dev/null 2>&1; do sleep 1; done
	@echo ">>> PostgreSQL container running on port $${POSTGRES_PORT:-5432}!"
	@echo ">>> Initializing PostgreSQL tables and semantic vector embeddings..."
	@$(PYTHON) -c "from hypoq.db import init_db, sync_all_embeddings; url = 'postgresql://hypoq:hypoq_secret@localhost:5432/hypoq'; init_db(url); n = sync_all_embeddings(url); print(f'>>> Successfully initialized PostgreSQL database ({url}) with {n} semantic embeddings!')"

install-postgres: install-pg
setup-pg: install-pg

install-sqlite:
	docker compose up -d hypoq-web
	@echo ">>> Initializing SQLite database schema and semantic vector embeddings..."
	@$(PYTHON) -c "from hypoq.db import init_db, sync_all_embeddings; init_db('hypoq.db'); n = sync_all_embeddings('hypoq.db'); print(f'>>> Successfully initialized SQLite database (hypoq.db) with {n} semantic embeddings!')"
	@echo ">>> SQLite database and hypoq web display active at http://127.0.0.1:$${PORT:-3030}!"

setup-sqlite: install-sqlite
