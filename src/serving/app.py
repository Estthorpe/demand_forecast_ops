# src/serving/app.py
"""
FastAPI inference server for demand-forecast-ops.

Endpoints:
    POST /forecast  — generate demand forecast with prediction intervals
    GET  /health    — liveness check (is the process running?)
    GET  /ready     — readiness check (is the model loaded?)
    GET  /metrics   — basic metrics in plain text (Prometheus-compatible)

"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.serving.predictor import predictor
from src.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
)

configure_logging()
logger = get_logger(__name__)

# Request counter for /metrics endpoint
# In production this would use Prometheus client library
# For now, a simple in-memory counter demonstrates the pattern
_request_count = 0
_error_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Code before yield runs at startup.
    Code after yield runs at shutdown.
    """
    # Startup
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

    # Shutdown
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
    Returns 503 if model is not loaded (startup failed or model missing).

    Load balancers should check /ready before routing traffic.
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

    The 90% interval means: with 90% confidence, actual demand
    will fall between lower_90 and upper_90 on that day.
    """
    global _request_count, _error_count
    _request_count += 1

    if not predictor.is_loaded:
        _error_count += 1
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is not ready to serve forecasts.",
        )

    logger.info(
        f"Forecast request: store={request.store_id}, start={request.start_date}, horizon={request.horizon_days}"
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

    return ForecastResponse(
        store_id=request.store_id,
        start_date=request.start_date,
        horizon_days=request.horizon_days,
        forecasts=forecasts,
        model_version=predictor.model_version,
        generated_at=datetime.now(UTC).isoformat(),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """
    Basic metrics endpoint in Prometheus text format.

    In production this would use the prometheus-client library.
    This minimal implementation demonstrates the pattern and
    satisfies the monitoring architecture requirement.
    """
    return "\n".join(
        [
            "# HELP forecast_requests_total Total forecast requests received",
            "# TYPE forecast_requests_total counter",
            f"forecast_requests_total {_request_count}",
            "# HELP forecast_errors_total Total forecast errors",
            "# TYPE forecast_errors_total counter",
            f"forecast_errors_total {_error_count}",
            "# HELP model_loaded Whether the model is currently loaded",
            "# TYPE model_loaded gauge",
            f"model_loaded {1 if predictor.is_loaded else 0}",
        ]
    )
