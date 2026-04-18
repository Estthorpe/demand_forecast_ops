# src/serving/app.py
"""
FastAPI inference server for demand-forecast-ops.

Endpoints:
    POST /forecast                   — generate demand forecast
    GET  /health                     — liveness check
    GET  /ready                      — readiness check
    GET  /metrics                    — Prometheus-compatible metrics
    GET  /monitoring/forecast        — latest forecast health report
    GET  /monitoring/drift           — latest drift report
    GET  /monitoring/trigger         — latest retraining trigger decision
    POST /monitoring/run-drift-check — trigger drift computation
    POST /genai/narrative            — generate plain-language narrative
    POST /agent/run                  — run replenishment agent
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.agents.replenishment_agent import ReplenishmentAgent, ReplenishmentPlan
from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.genai.narrative_generator import NarrativeResult, generate_narrative
from src.monitoring.drift import (
    DriftReport,
    compute_drift,
    load_reference_distribution,
    save_drift_report,
)
from src.monitoring.forecast_tracker import (
    ForecastHealthReport,
    track_forecast_health,
)
from src.monitoring.retraining_trigger import (
    DEFAULT_TRIGGER_LOG_PATH,
    TriggerReport,
    append_trigger_log,
    evaluate_trigger,
)
from src.serving.predictor import predictor
from src.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
)

configure_logging()
logger = get_logger(__name__)

# ── In-memory metrics store ─────────────────────────────────────────────────
_request_count: int = 0
_error_count: int = 0
_total_latency_ms: float = 0.0
_prediction_buffer: list[float] = []

_latest_drift_report: DriftReport | None = None
_latest_forecast_report: ForecastHealthReport | None = None
_latest_trigger_report: TriggerReport | None = None

MONITORING_OUTPUT_DIR = Path("logs/monitoring")
REFERENCE_DISTRIBUTION_PATH = Path("models/reference_distribution.parquet")


# ── Request latency middleware ──────────────────────────────────────────────
class LatencyMiddleware(BaseHTTPMiddleware):
    """Track request latency for all endpoints."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        global _total_latency_ms

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        _total_latency_ms += elapsed_ms
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.debug(
            f"{request.method} {request.url.path} " f"completed in {elapsed_ms:.2f}ms"
        )

        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting demand-forecast-ops serving layer...")
    try:
        predictor.load_model()
        logger.info("Model loaded successfully — server is ready")
    except FileNotFoundError as e:
        logger.warning(
            f"Model artefact not found: {e}. "
            "Server will start but /ready will return 503."
        )
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")

    yield

    logger.info("Shutting down demand-forecast-ops serving layer")


app = FastAPI(
    title="demand-forecast-ops",
    description=(
        "Production demand forecasting API for retail supply chain. "
        "Returns point forecasts and 90% prediction intervals."
    ),
    version=settings.project_version,
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)


# ── Core endpoints ──────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — always returns 200 if process is alive."""
    return HealthResponse(
        status="ok",
        version=settings.project_version,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness check — returns 200 if model is loaded."""
    if predictor.is_loaded:
        return ReadyResponse(
            status="ready",
            model_loaded=True,
        )
    raise HTTPException(
        status_code=503,
        detail="Model not loaded. Run python scripts/train.py first.",
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Generate demand forecast with 90% prediction intervals.

    Accepts a store ID, start date, and horizon length.
    Returns daily point forecasts with lower and upper bounds.
    Also updates the in-memory prediction buffer for monitoring.
    """
    global _request_count, _error_count, _prediction_buffer
    global _latest_forecast_report, _latest_trigger_report

    _request_count += 1

    if not predictor.is_loaded:
        _error_count += 1
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is not ready to serve forecasts.",
        )

    logger.info(
        f"Forecast request: store={request.store_id}, "
        f"start={request.start_date}, "
        f"horizon={request.horizon_days}"
    )

    try:
        forecasts = predictor.predict(
            store_id=request.store_id,
            start_date=request.start_date,
            horizon_days=request.horizon_days,
            promo=request.promo,
            school_holiday=request.school_holiday,
            state_holiday=request.state_holiday,
        )
    except Exception as e:
        _error_count += 1
        logger.error(f"Forecast failed for store {request.store_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}",
        )

    # ── Update prediction buffer for monitoring ─────────────────────────────
    point_forecasts = [f.point_forecast for f in forecasts]
    _prediction_buffer.extend(point_forecasts)

    if len(_prediction_buffer) > 1000:
        _prediction_buffer = _prediction_buffer[-1000:]

    # ── Run forecast health check ───────────────────────────────────────────
    try:
        _latest_forecast_report = track_forecast_health(
            predictions=np.array(_prediction_buffer),
            period=datetime.now(UTC).strftime("%Y-%m-%d"),
            lower_bounds=np.array([f.lower_90 for f in forecasts]),
            upper_bounds=np.array([f.upper_90 for f in forecasts]),
        )

        if _latest_drift_report is not None:
            _latest_trigger_report = evaluate_trigger(
                drift_report=_latest_drift_report,
                forecast_report=_latest_forecast_report,
            )
            append_trigger_log(
                _latest_trigger_report,
                DEFAULT_TRIGGER_LOG_PATH,
            )

            if _latest_trigger_report.should_retrain:
                logger.warning(
                    f"Retraining flag raised: "
                    f"{_latest_trigger_report.recommended_action}"
                )

    except Exception as e:
        logger.error(f"Monitoring update failed (non-fatal): {e}")

    return ForecastResponse(
        store_id=request.store_id,
        start_date=request.start_date,
        horizon_days=request.horizon_days,
        forecasts=forecasts,
        model_version=predictor.model_version,
        generated_at=datetime.now(UTC).isoformat(),
    )


# ── Metrics endpoint ────────────────────────────────────────────────────────


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-compatible metrics endpoint."""
    avg_latency = _total_latency_ms / _request_count if _request_count > 0 else 0.0
    error_rate = _error_count / _request_count if _request_count > 0 else 0.0
    mean_forecast = float(np.mean(_prediction_buffer)) if _prediction_buffer else 0.0
    retraining_flag = (
        1
        if _latest_trigger_report is not None and _latest_trigger_report.should_retrain
        else 0
    )
    drift_flag = (
        1
        if _latest_drift_report is not None and _latest_drift_report.drift_detected
        else 0
    )

    lines = [
        "# HELP forecast_requests_total Total forecast requests received",
        "# TYPE forecast_requests_total counter",
        f"forecast_requests_total {_request_count}",
        "",
        "# HELP forecast_errors_total Total forecast errors",
        "# TYPE forecast_errors_total counter",
        f"forecast_errors_total {_error_count}",
        "",
        "# HELP forecast_error_rate Current error rate",
        "# TYPE forecast_error_rate gauge",
        f"forecast_error_rate {error_rate:.4f}",
        "",
        "# HELP forecast_avg_latency_ms Average request latency in ms",
        "# TYPE forecast_avg_latency_ms gauge",
        f"forecast_avg_latency_ms {avg_latency:.2f}",
        "",
        "# HELP model_loaded Whether the model is currently loaded",
        "# TYPE model_loaded gauge",
        f"model_loaded {1 if predictor.is_loaded else 0}",
        "",
        "# HELP mean_forecast_value Mean of recent point forecasts",
        "# TYPE mean_forecast_value gauge",
        f"mean_forecast_value {mean_forecast:.2f}",
        "",
        "# HELP drift_detected Whether input drift has been detected",
        "# TYPE drift_detected gauge",
        f"drift_detected {drift_flag}",
        "",
        "# HELP retraining_flag Whether retraining has been triggered",
        "# TYPE retraining_flag gauge",
        f"retraining_flag {retraining_flag}",
    ]

    return "\n".join(lines)


# ── Monitoring endpoints ────────────────────────────────────────────────────


@app.get("/monitoring/forecast")
async def monitoring_forecast() -> JSONResponse:
    """Latest forecast health report."""
    if _latest_forecast_report is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_data",
                "message": "No forecast health data yet. "
                "Call /forecast at least once.",
            },
        )
    return JSONResponse(content=_latest_forecast_report.to_dict())


@app.get("/monitoring/drift")
async def monitoring_drift() -> JSONResponse:
    """Latest drift report."""
    if _latest_drift_report is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_data",
                "message": "No drift report yet. "
                "Call /monitoring/run-drift-check first.",
            },
        )
    return JSONResponse(content=_latest_drift_report.to_dict())


@app.get("/monitoring/trigger")
async def monitoring_trigger() -> JSONResponse:
    """Latest retraining trigger decision."""
    if _latest_trigger_report is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_data",
                "message": "No trigger report yet.",
            },
        )
    return JSONResponse(content=_latest_trigger_report.to_dict())


@app.post("/monitoring/run-drift-check")
async def run_drift_check() -> JSONResponse:
    """Trigger a drift computation against the reference distribution."""
    global _latest_drift_report, _latest_trigger_report

    if not REFERENCE_DISTRIBUTION_PATH.exists():
        return JSONResponse(
            status_code=200,
            content={
                "status": "reference_missing",
                "message": (
                    f"Reference distribution not found at "
                    f"{REFERENCE_DISTRIBUTION_PATH}. "
                    "Run scripts/compute_reference_distribution.py first."
                ),
            },
        )

    if len(_prediction_buffer) < 10:
        return JSONResponse(
            status_code=200,
            content={
                "status": "insufficient_data",
                "message": (
                    "Need at least 10 predictions before drift can be computed. "
                    f"Current buffer: {len(_prediction_buffer)} predictions."
                ),
            },
        )

    try:
        import pandas as pd

        reference_df = load_reference_distribution(REFERENCE_DISTRIBUTION_PATH)
        current_df = pd.DataFrame({"point_forecast": _prediction_buffer})

        _latest_drift_report = compute_drift(
            reference_df=reference_df,
            current_df=current_df,
            reference_period="training_2013_2015",
            current_period=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

        save_drift_report(
            _latest_drift_report,
            MONITORING_OUTPUT_DIR / "latest_drift_report.json",
        )

        if _latest_forecast_report is not None:
            _latest_trigger_report = evaluate_trigger(
                drift_report=_latest_drift_report,
                forecast_report=_latest_forecast_report,
            )
            append_trigger_log(_latest_trigger_report, DEFAULT_TRIGGER_LOG_PATH)

        return JSONResponse(content=_latest_drift_report.to_dict())

    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Drift check failed: {str(e)}",
        )


# ── GenAI endpoint ──────────────────────────────────────────────────────────


@app.post("/genai/narrative")
async def genai_narrative(request: dict) -> JSONResponse:  # type: ignore[type-arg]
    """
    Generate a plain-language narrative from structured forecast
    and monitoring metadata.

    Accepts a JSON dict containing any combination of:
      - store_id, horizon_days, mean_forecast (forecast data)
      - drift_detected, max_psi (drift data)
      - should_retrain, recommended_action (trigger data)

    Optional field:
      - template: "forecast", "monitoring", or "combined" (default)

    Returns a grounded narrative — only data in the input is used.
    Falls back to a template-based narrative if LLM is unavailable.
    """
    template_name = request.pop("template", "combined")

    try:
        result: NarrativeResult = generate_narrative(
            metadata=request,
            template_name=str(template_name),
        )
        return JSONResponse(content=result.to_dict())

    except Exception as e:
        logger.error(f"Narrative endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Narrative generation failed: {str(e)}",
        )


# ── Agent endpoint ──────────────────────────────────────────────────────────


@app.post("/agent/run")
async def agent_run(request: dict) -> JSONResponse:  # type: ignore[type-arg]
    """
    Run the Replenishment Agent for a store.

    Orchestrates the full workflow:
      1. Health check
      2. Demand forecast
      3. Monitoring trigger check
      4. Narrative generation
      5. Replenishment plan

    Required fields:
      - store_id: int
      - start_date: str (YYYY-MM-DD)

    Optional fields:
      - horizon_days: int (default 7)
      - promo: int (default 0)

    Returns a ReplenishmentPlan with full audit trail.
    Agent never places orders — produces plan for review only.
    """
    try:
        store_id = int(request.get("store_id", 1))
        start_date = date.fromisoformat(
            str(request.get("start_date", date.today().isoformat()))
        )
        horizon_days = int(request.get("horizon_days", 7))
        promo = int(request.get("promo", 0))

        agent = ReplenishmentAgent(base_url="http://localhost:8000")

        # Run agent in thread executor to avoid blocking the async event loop.
        # The agent uses synchronous httpx — calling itself from within the
        # same server requires a separate thread to prevent deadlock on Windows.
        loop = asyncio.get_event_loop()
        plan: ReplenishmentPlan = await loop.run_in_executor(
            None,
            partial(
                agent.run,
                store_id=store_id,
                start_date=start_date,
                horizon_days=horizon_days,
                promo=promo,
            ),
        )

        return JSONResponse(content=plan.to_dict())

    except Exception as e:
        logger.error(f"Agent endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent run failed: {str(e)}",
        )
