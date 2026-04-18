"""
FastAPI inference server for demand-forecast-ops.

Endpoints:
    POST /forecast  — generate demand forecast with prediction intervals
    GET  /health    — liveness check (is the process running?)
    GET  /ready     — readiness check (is the model loaded?)
    GET  /metrics   — Prometheus-compatible metrics
    GET  /monitoring/drift     — latest drift report
    GET  /monitoring/forecast  — latest forecast health report
    GET  /monitoring/trigger   — latest retraining trigger decision
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
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
# In production this would use prometheus-client library.
# This implementation demonstrates the pattern correctly.
_request_count: int = 0
_error_count: int = 0
_total_latency_ms: float = 0.0
_prediction_buffer: list[float] = []  # Rolling buffer of recent predictions

# Monitoring report cache — updated on each /forecast call
_latest_drift_report: DriftReport | None = None
_latest_forecast_report: ForecastHealthReport | None = None
_latest_trigger_report: TriggerReport | None = None

# Paths for monitoring outputs
MONITORING_OUTPUT_DIR = Path("logs/monitoring")
REFERENCE_DISTRIBUTION_PATH = Path("models/reference_distribution.parquet")


# ── Request latency middleware ──────────────────────────────────────────────
class LatencyMiddleware(BaseHTTPMiddleware):
    """
    Track request latency for all endpoints.

    Adds X-Response-Time header to all responses.
    Updates global latency counter for /metrics endpoint.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        global _total_latency_ms

        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        _total_latency_ms += elapsed_ms
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.debug(
            f"{request.method} {request.url.path} " f"completed in {elapsed_ms:.2f}ms"
        )

        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Code before yield runs at startup.
    Code after yield runs at shutdown.
    """
    logger.info("Starting demand-forecast-ops serving layer...")
    try:
        predictor.load_model()
        logger.info("Model loaded successfully — server is ready")
    except FileNotFoundError as e:
        logger.warning(
            f"Model artefact not found: {e}. "
            "Server will start but /ready will return 503. "
            "Run python scripts/train.py to generate the model."
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

# Register middleware
app.add_middleware(LatencyMiddleware)


# ── Core endpoints ──────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness check — is the process running?

    Always returns 200 if the process is alive.
    Does not check model state — use /ready for that.
    """
    return HealthResponse(
        status="ok",
        version=settings.project_version,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """
    Readiness check — is the model loaded and ready to serve?

    Returns 200 if model is loaded.
    Returns 503 if model is not loaded.
    """
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

    # Keep buffer bounded — last 1000 predictions
    if len(_prediction_buffer) > 1000:
        _prediction_buffer = _prediction_buffer[-1000:]

    # ── Run forecast health check on every request ──────────────────────────
    try:
        _latest_forecast_report = track_forecast_health(
            predictions=np.array(_prediction_buffer),
            period=datetime.now(UTC).strftime("%Y-%m-%d"),
            lower_bounds=np.array([f.lower_90 for f in forecasts]),
            upper_bounds=np.array([f.upper_90 for f in forecasts]),
        )

        # Run trigger evaluation if drift report exists
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
        # Monitoring must never break serving
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
    """
    Prometheus-compatible metrics endpoint.

    Exposes request counts, error rates, latency,
    and model health indicators.
    """
    avg_latency = _total_latency_ms / _request_count if _request_count > 0 else 0.0

    error_rate = _error_count / _request_count if _request_count > 0 else 0.0

    # Forecast distribution stats if available
    mean_forecast = 0.0
    if _prediction_buffer:
        mean_forecast = float(np.mean(_prediction_buffer))

    retraining_flag = 0
    if _latest_trigger_report is not None:
        retraining_flag = 1 if _latest_trigger_report.should_retrain else 0

    drift_flag = 0
    if _latest_drift_report is not None:
        drift_flag = 1 if _latest_drift_report.drift_detected else 0

    lines = [
        "# HELP forecast_requests_total Total forecast requests received",
        "# TYPE forecast_requests_total counter",
        f"forecast_requests_total {_request_count}",
        "",
        "# HELP forecast_errors_total Total forecast errors",
        "# TYPE forecast_errors_total counter",
        f"forecast_errors_total {_error_count}",
        "",
        "# HELP forecast_error_rate Current error rate (errors/requests)",
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
    """
    Latest forecast health report.

    Returns distribution stats and error metrics
    from the most recent monitoring cycle.
    """
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
    """
    Latest drift report.

    Returns PSI values per feature and overall drift flag.
    Drift is computed by calling /monitoring/run-drift-check.
    """
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
    """
    Latest retraining trigger decision.

    Returns should_retrain flag and full audit trail
    of which rules fired and why.
    """
    if _latest_trigger_report is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_data",
                "message": "No trigger report yet. "
                "Make forecast requests to generate one.",
            },
        )
    return JSONResponse(content=_latest_trigger_report.to_dict())


@app.post("/monitoring/run-drift-check")
async def run_drift_check() -> JSONResponse:
    """
    Trigger a drift computation against the reference distribution.

    Loads reference distribution from models/reference_distribution.parquet.
    Computes PSI against recent prediction buffer features.
    Updates the in-memory drift report and trigger report.

    Returns the full drift report.
    """
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
        reference_df = load_reference_distribution(REFERENCE_DISTRIBUTION_PATH)

        # Build current DataFrame from prediction buffer
        import pandas as pd

        current_df = pd.DataFrame(
            {
                "point_forecast": _prediction_buffer,
            }
        )

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

        # Re-run trigger if forecast report exists
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
