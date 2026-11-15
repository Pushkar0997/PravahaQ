.PHONY: help install install-quantum test test-unit test-integration test-property lint format api frontend benchmark paper docker-up docker-down clean

PYTHON ?= python3
PIP ?= pip

help:
	@echo "======================================================================"
	@echo "  PravahaQ v0.1.0 — Developer & Research Automation CLI               "
	@echo "======================================================================"
	@echo "  make install          Install core package + dev dependencies"
	@echo "  make install-quantum  Install Qiskit, PennyLane, and Cirq adapters"
	@echo "  make test             Run complete 4-tier test suite (Unit/Int/Prop)"
	@echo "  make benchmark        Execute full multi-algorithm benchmark suite"
	@echo "  make paper            Generate 8 vector figures & compile paper PDF"
	@echo "  make api              Launch FastAPI control plane on port 8000"
	@echo "  make frontend         Launch Quantum Obsidian UI on port 5173"
	@echo "  make docker-up        Spin up full-stack Docker Compose environment"
	@echo "  make lint             Run Ruff & MyPy static analysis"
	@echo "  make clean            Remove build/cache artifacts"
	@echo "======================================================================"

install:
	$(PIP) install -e ".[dev]"

install-quantum:
	$(PIP) install -e ".[dev,quantum]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-property:
	pytest tests/property/ -v

lint:
	ruff check src/ api/ tests/
	mypy src/pravaha_q

format:
	ruff format src/ api/ tests/

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev

benchmark:
	$(PYTHON) experiments/run_full_frontier_benchmarks.py

paper:
	$(PYTHON) paper/build_paper_and_figures.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
