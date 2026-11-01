# Healthcare Knowledge Navigator — every action is a target, and CI runs the same ones.
# Targets that call Claude say so and refuse to spend without YES=1.

PY      := backend/.venv/bin/python
HKN     := cd backend && .venv/bin/python -m hkn.cli
EVAL    := cd backend && .venv/bin/python -m
NPM     := cd frontend && npm
SPEND   := $(if $(YES),--yes,)

export HF_HOME := $(CURDIR)/.hf_cache
export TOKENIZERS_PARALLELISM := false

.PHONY: help setup db-up db-down migrate ingest-seed ingest-pmc contextualize contextualize-collect \
        api web dev test test-integration lint format eval-retrieval eval-retrieval-plan tune-gate \
        eval-answers-extractive eval-answers label review calibrate synthesize

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-26s %s\n", $$1, $$2}'

setup:  ## install backend (uv) and frontend (npm) dependencies
	cd backend && uv sync
	$(NPM) ci

db-up:  ## start Postgres + pgvector on :5434 and wait until healthy
	docker compose up -d db
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' hkn-db)" = "healthy" ]; do sleep 1; done

db-down:  ## stop Postgres (data is kept in the hkn_pg volume)
	docker compose down

migrate:  ## apply database migrations
	cd backend && .venv/bin/python -m alembic upgrade head

ingest-seed:  ## fetch, parse, chunk and embed the 12 seed documents (free)
	$(HKN) ingest seed

ingest-pmc:  ## ingest ~300 CC BY / CC0 PubMed Central articles (free; set HKN_NCBI_EMAIL)
	$(HKN) ingest pmc --per-topic 75

contextualize:  ## Claude chunk contexts via Batches — prints an estimate; YES=1 submits (paid)
	$(HKN) ingest contextualize --scope guidelines $(SPEND)

contextualize-collect:  ## wait for submitted context batches, store them, re-embed
	$(HKN) ingest contextualize --collect

api:  ## FastAPI on :8002
	cd backend && .venv/bin/python -m uvicorn hkn.api.main:app --port 8002 --reload

web:  ## Next.js on :3000
	$(NPM) run dev

dev:  ## API and web together (Ctrl-C stops both)
	@trap 'kill 0' INT TERM; $(MAKE) api & $(MAKE) web & wait

test:  ## unit tests, backend and frontend (offline)
	cd backend && .venv/bin/python -m pytest -q
	$(NPM) test -- --run

test-integration:  ## SQL-level tests against the hkn_test database (needs db-up)
	cd backend && .venv/bin/python -m pytest -q -m integration

lint:  ## ruff, ruff format --check, pyright, eslint, tsc
	cd backend && .venv/bin/python -m ruff check src evals tests alembic
	cd backend && .venv/bin/python -m ruff format --check src evals tests alembic
	cd backend && .venv/bin/python -m pyright src evals
	$(NPM) run lint
	$(NPM) run typecheck

format:  ## apply ruff formatting
	cd backend && .venv/bin/python -m ruff format src evals tests alembic

eval-retrieval:  ## retrieval ablation ladder, free rungs (local models)
	$(EVAL) evals.run_retrieval

eval-retrieval-plan:  ## full ladder incl. query plan + HyDE (paid once; plans are cached)
	$(EVAL) evals.run_retrieval --with-plan $(SPEND)

tune-gate:  ## fit the retrieval gate threshold on dev, report on test (free)
	$(EVAL) evals.tune_gate

eval-answers-extractive:  ## answer eval with the extractive baseline (free)
	$(EVAL) evals.run_answers --mode extractive --no-plan

eval-answers:  ## answer eval with Claude, 4 configurations (paid; prints estimate)
	$(EVAL) evals.run_answers --mode claude $(SPEND)

label:  ## label the latest answer run with the Claude judge (paid)
	$(EVAL) evals.label --labeler claude $(SPEND)

review:  ## review labels by hand; prints judge–human Cohen's κ
	$(EVAL) evals.review --against claude

calibrate:  ## fit reliability weights + bands; writes confidence_weights.json
	$(EVAL) evals.calibrate

synthesize:  ## synthesize eval questions (paid); then: cd backend && python -m evals.synthesize --review
	$(EVAL) evals.synthesize --n 60 $(SPEND)
