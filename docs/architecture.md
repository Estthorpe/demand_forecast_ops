# System Architecture — demand-forecast-ops

## Overview

demand-forecast-ops is a production-grade retail demand forecasting system
organised as a 7-stage AI engineering pipeline. Each stage produces artifacts
consumed by the next.

## Component Diagram

┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│          HTTP requests → FastAPI (src/serving/app.py)        │
└─────────────────────┬───────────────────────────────────────┘
│
┌────────────▼────────────┐
│     Serving Layer        │
│   src/serving/           │
│   - app.py (FastAPI)     │
│   - predictor.py         │
│   - schemas.py           │
└──┬─────────┬────────────┘
│         │
┌───────▼──┐  ┌───▼──────────┐
│  Model   │  │  Monitoring  │
│  Layer   │  │  Layer       │
│ src/     │  │ src/         │
│ models/  │  │ monitoring/  │
└───────┬──┘  └───┬──────────┘
│         │
┌───────▼─────────▼──────────┐
│       Feature Layer         │
│       src/features/         │
└───────────────┬─────────────┘
│
┌───────────────▼─────────────┐
│       Ingestion Layer        │
│       src/ingestion/         │
└─────────────────────────────┘
┌─────────────────────────────┐
│       GenAI Layer            │
│       src/genai/             │
│   Calls Anthropic API        │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│       Agent Layer            │
│       src/agents/            │
│   Calls Serving via HTTP     │
│   Produces ReplenishmentPlan │
└─────────────────────────────┘

## Data Flow

### Training Flow
Raw CSV (Kaggle)
→ src/ingestion/loader.py (validation)
→ src/ingestion/contracts.py (Pydantic contracts)
→ src/features/pipeline.py (calendar + lag + rolling)
→ scripts/train.py (LightGBM fit)
→ models/lgbm_forecaster.joblib (DVC-tracked)

### Inference Flow
POST /forecast
→ src/serving/schemas.py (request validation)
→ src/serving/predictor.py (feature build + model predict)
→ src/monitoring/forecast_tracker.py (health check)
→ src/serving/schemas.py (response serialisation)
→ JSON response

### Agent Flow
POST /agent/run
→ src/agents/replenishment_agent.py
→ tool: GET /ready (health check)
→ tool: POST /forecast (demand forecast)
→ tool: GET /monitoring/trigger (retraining signal)
→ tool: POST /genai/narrative (stakeholder narrative)
→ ReplenishmentPlan (audit logged to logs/agent_audit.jsonl)


## Key Design Decisions

### Train-Serve Consistency
`src/features/pipeline.py` is the single entry point for all feature
computation. Both training and serving call the same pipeline. This
prevents training-serving skew.

**Exception (T-012):** Lag and rolling features use groupby.transform
which deadlocks in uvicorn's async event loop on Windows. At inference
time, these features default to 0. This is a known trade-off documented
in risk_log.md.

### Evaluation-as-Tests
Model quality gates are enforced as pytest tests in CI. A model that
does not beat the seasonal naive baseline by the required threshold
will fail CI. This prevents silent model degradation.

### Monitoring Architecture
Monitoring modules are strictly separated by responsibility:
- `drift.py` — computes PSI, produces DriftReport
- `forecast_tracker.py` — tracks prediction distribution, produces ForecastHealthReport
- `retraining_trigger.py` — combines signals, produces TriggerReport

No module triggers retraining directly. All produce reports consumed
by the agent layer.

### Agent Constraints
The Replenishment Agent operates within a strictly constrained action space:
- All model access is via HTTP tool calls — never direct imports
- Replenishment quantities are clipped to [0, 50,000] guardrails
- Every decision is written to an append-only audit log
- The agent produces plans for review — it never places orders

### GenAI Grounding
The narrative generator receives only the metadata dict passed to it.
Grounding is enforced at the prompt level — the model is instructed
to use only provided figures. A structured fallback runs without
the LLM if the API is unavailable.

## Infrastructure

| Component | Technology |
|-----------|-----------|
| API server | FastAPI + uvicorn |
| Model artefacts | DVC → DagsHub |
| Experiment tracking | MLflow → DagsHub |
| CI/CD | GitHub Actions |
| Deployment | Railway.app |
| Containerisation | Docker (multi-stage) |
| Secrets | Railway environment variables |

## Port and Network

- Local development: `http://localhost:8000`
- Production: `https://demandforecastops-production.up.railway.app`
- Internal Railway network: `demand-forecast-ops.railway.internal`
