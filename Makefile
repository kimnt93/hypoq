# ==============================================================================
# hypoq — Hypothesis x Quant: autonomous research harness and free, real data lake
# Repository: https://github.com/kimnt93/hypoq.git
# ==============================================================================

# optional settings, see .env.example
-include .env
.EXPORT_ALL_VARIABLES:

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo $$(pwd)/.venv/bin/python3; elif [ -f ../.venv/bin/python3 ]; then echo $$(cd .. && pwd)/.venv/bin/python3; else which python3; fi)
ANYMD ?= $(shell which anymd || echo "/home/kim/.bun/bin/anymd")
UV ?= $(shell which uv || echo "/home/kim/.local/bin/uv")

.PHONY: all install download verify audit seed test api parse-doc help clean start stop install-pg install-postgres install-sqlite setup-pg setup-sqlite db-import db-migrate db-export

# Default target downloads and builds all datasets
all: download

help:
	@echo "========================================================================"
	@echo " hypoq: Autonomous Quantitative Research Lab & Empirical Modeling"
	@echo "========================================================================"
	@echo " make install          Install host research stack (.venv) & start default SQLite stack"
	@echo " make download         Download, parse with anymd, and build all datasets"
	@echo " make verify           Data-quality audit of all Parquet files (duplicates, nulls, OHLC, staleness)"
	@echo " make audit            Research integrity audit (code, results JSON, PDF and database must agree)"
	@echo " make seed             Re-run every researches/*/code/run_pipeline.py and refresh the database"
	@echo " make db-import        Load researches/*/results/hypothesis_record.json into the (git-ignored) database and embed them"
	@echo " make db-migrate       Re-embed all hypotheses after changing EMBEDDING_PROVIDER / EMBEDDING_MODEL (add FORCE=1 to redo all)"
	@echo " make db-export        Back-fill hypothesis_record.json files from the database
	@echo " make test             Run the pytest suite"
	@echo " make api              Start the JSON API + dashboard (api.py) on http://127.0.0.1:8080"
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
	@echo ">>> Starting complete free-data ingestion pipeline..."
	$(PYTHON) scripts/download_all.py
	@echo ">>> All datasets successfully downloaded and verified!"

verify:
	@echo ">>> Validating Parquet Datasets Integrity..."
	$(PYTHON) scripts/verify_datasets.py

audit:
	$(PYTHON) run.py --audit

seed:
	$(PYTHON) scripts/seed_database.py

db-import:
	$(PYTHON) scripts/db_import.py

db-migrate:
	$(PYTHON) scripts/db_migrate_embeddings.py $(if $(FORCE),--force,)

db-export:
	$(PYTHON) scripts/db_export_records.py

test:
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) run.py --web --port 8080

parse-doc:
	@if [ -z "$(FILE)" ]; then echo "Usage: make parse-doc FILE=<path_or_url> [PAGES=1-5]"; exit 1; fi
	$(ANYMD) $(FILE) $(if $(PAGES),-p $(PAGES),)

clean:
	@echo ">>> Cleaning temporary caches and artifacts..."
	@rm -rf /tmp/anymd* .pytest_cache __pycache__
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
	@until docker compose exec -T postgres pg_isready -U $${POSTGRES_USER:-hypoq} -d $${POSTGRES_DB:-hypoq} >/dev/null 2>&1; do sleep 1; done
	@echo ">>> PostgreSQL container running on port $${POSTGRES_PORT:-5432}!"
	@echo ">>> Importing the example studies and creating their embeddings..."
	DATABASE_URL=$${DATABASE_URL:-postgresql://$${POSTGRES_USER:-hypoq}:$${POSTGRES_PASSWORD:-hypoq_secret}@localhost:$${POSTGRES_PORT:-5432}/$${POSTGRES_DB:-hypoq}} $(PYTHON) scripts/db_import.py

install-postgres: install-pg
setup-pg: install-pg

install-sqlite:
	@echo ">>> Importing the example studies into the local SQLite database (hypoq.db) and creating their embeddings..."
	$(PYTHON) scripts/db_import.py --db hypoq.db
	docker compose up -d hypoq-web
	@echo ">>> SQLite database and hypoq web display active at http://127.0.0.1:$${PORT:-3030}!"

setup-sqlite: install-sqlite
