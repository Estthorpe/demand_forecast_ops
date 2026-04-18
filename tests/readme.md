# demand-forecast-ops

Production-grade retail demand forecasting system. Built as P5 of 8 in an AI Engineering Portfolio targeting AI Engineer / ML Engineer roles in 2026.

**Live endpoint:** `https://demandforecastops-production.up.railway.app`

---

## What This System Does

Forecasts daily retail demand for 1,115 Rossmann stores across a configurable horizon. Returns point forecasts with 90% prediction intervals. Monitors for data drift and model degradation. Generates plain-language stakeholder narratives. Orchestrates replenishment planning via an autonomous agent.

---

## Engineering Lifecycle

This project follows a mandatory 7-stage AI engineering lifecycle:

| Stage | What Was Built |
|-------|---------------|
| 1 — Template | Typed config, structured logging, pre-commit hooks, CI pipeline, Docker |
| 2 — Ingestion & Data Contracts | Rossmann dataset ingestion, Pydantic schema validation, DVC versioning |
| 3 — Evaluation-as-Tests | Walk-forward validation, MASE quality gates, regression tests |
| 4 — Serving | FastAPI inference endpoint, typed schemas, containerised deployment |
| 5 — Monitoring | PSI drift detection, forecast health tracking, retraining trigger logic |
| 6 — GenAI Integration | LLM narrative generator, versioned prompts, grounding constraints |
| 7 — Agentic Layer | Replenishment agent, tool-based orchestration, audit log |

---

## Repository Structure


---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check — model loaded? |
| POST | `/forecast` | Generate demand forecast with 90% intervals |
| GET | `/metrics` | Prometheus-compatible metrics |
| GET | `/monitoring/forecast` | Latest forecast health report |
| GET | `/monitoring/drift` | Latest PSI drift report |
| GET | `/monitoring/trigger` | Latest retraining trigger decision |
| POST | `/monitoring/run-drift-check` | Trigger drift computation |
| POST | `/genai/narrative` | Generate stakeholder narrative |
| POST | `/agent/run` | Run replenishment agent |

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Estthorpe/demand_forecast_ops
cd demand_forecast_ops
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Train model
python scripts/train.py

# Run evaluation gates
python scripts/evaluate.py

# Start server
uvicorn src.serving.app:app --port 8000
```

---

## Forecast Request Example

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "start_date": "2015-07-01",
    "horizon_days": 7
  }'
```

## Agent Run Example

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "start_date": "2015-07-01",
    "horizon_days": 7
  }'
```

---

## Model Performance

| Metric | Baseline (Seasonal Naive) | LightGBM |
|--------|--------------------------|----------|
| MASE | 1.2956 | 1.0338 |
| Improvement | — | 20.21% |

Evaluated using walk-forward expanding window validation across 1,115 stores.

---

## Dataset

**Rossmann Store Sales** — 982,644 records, 1,115 stores, 2013–2015.
Source: [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
Tracked via DVC on DagsHub: https://dagshub.com/Estthorpe/demand_forecast_ops

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Forecasting | LightGBM, scikit-learn |
| Serving | FastAPI, uvicorn, httpx |
| Data validation | Pydantic |
| Experiment tracking | MLflow, DagsHub |
| Data versioning | DVC |
| Monitoring | Custom PSI, forecast tracker |
| GenAI | Anthropic Claude Haiku |
| CI/CD | GitHub Actions |
| Deployment | Railway.app |
| Code quality | ruff, mypy, pre-commit |

---

## Key Engineering Decisions

| ID | Decision | Reason |
|----|----------|--------|
| T-002 | Railway.app for deployment | Docker blocked on corporate machine |
| T-003 | Direct LightGBM over recursive | No error accumulation across horizon |
| T-006 | Hand-rolled agent before LangGraph | Understanding before abstraction |
| T-012 | Calendar features only at inference | groupby.transform deadlocks in uvicorn async on Windows |
| T-013 | DVC pull at runtime not build time | Railway free tier build secrets limitation |

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — System design and component relationships
- [`docs/model_card.md`](docs/model_card.md) — Model facts, limitations, intended use
- [`docs/runbook.md`](docs/runbook.md) — Operational procedures and monitoring response
- [`docs/risk_log.md`](docs/risk_log.md) — Engineering trade-offs and lessons learned
- [`docs/validation_strategy.md`](docs/validation_strategy.md) — Evaluation methodology
