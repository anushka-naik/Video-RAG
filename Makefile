PYTHON ?= .venv/bin/python

.PHONY: install install-all install-server up down dev worker test test-all eval eval-perception research-index demo lint typecheck typecheck-all

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e "backend[dev,research,demo]"

install-all:
	$(PYTHON) -m pip install -e "backend[dev,research,demo,server]"

install-server:
	$(PYTHON) -m pip install -e "backend[server]"

up:
	docker compose up -d

down:
	docker compose down

dev:
	$(PYTHON) -m uvicorn app.main:app --app-dir backend --reload

worker:
	cd backend && ../.venv/bin/arq app.workers.ingestion.WorkerSettings

test:
	cd backend && ../$(PYTHON) -m pytest tests/test_evaluation_metrics.py tests/test_perception_test.py tests/test_research_experiment.py tests/test_research_retrieval.py tests/test_research_store.py

test-all:
	$(PYTHON) -m pytest backend/tests

eval:
	cd backend && ../$(PYTHON) -m app.evaluation.runner $(ARGS)

research-index:
	PYTHONPATH=backend $(PYTHON) -m app.research.cli index-perception-test $(ARGS)

eval-perception:
	PYTHONPATH=backend $(PYTHON) -m app.research.cli evaluate-perception-test $(ARGS)

demo:
	PYTHONPATH=backend $(PYTHON) -m streamlit run backend/app/research/demo.py

lint:
	$(PYTHON) -m ruff check backend/app backend/tests
	$(PYTHON) -m ruff format --check backend/app backend/tests

typecheck:
	cd backend && ../$(PYTHON) -m mypy --strict app/research app/evaluation

typecheck-all:
	cd backend && ../$(PYTHON) -m mypy --strict app
