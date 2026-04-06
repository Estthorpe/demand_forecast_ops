"""
Request and response schemas for the forecast API.
The Pydantic models define the API contract
They serve three purposes:
1. Input validation - ensuring that incoming requests have the correct structure and data types and rejects malformed requests with clear error messages
2. Output serialization - defining the structure of the API responses, ensuring that the output is consistent and can be easily consumed by clients
3. Documentation - automatically generating OpenAPI documentation from these schemas"""

from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request body for POST /forecast endpoint
    The client specifies which store and date to forecast for, and the API returns the predicted sales for that store and date"""

    store_id: int = Field(
        ge=1,
        le=1115,
        description="Rossmann store identifier (1-1115)",
        examples=[1],
    )
    start_date: date = Field(
        description="First date to forecast (YYYY-MM-DD)",
        examples=["2024-01-01"],
    )
    horizon_days: int = Field(
        default=14,
        ge=1,
        le=30,
        description="Number of days to forecast (1-30). Default is 14",
    )
    promo: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether the store has an active promotion during the forecast period (0 or 1). Default is 0",
    )
    school_holiday: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether the store is affected by school holidays during the forecast period (0 or 1). Default is 0",
    )
    state_holiday: str = Field(
        default=0,
        description="State holiday indicator: '0'=none, 'a'=public, 'b'=Easter, 'c'=Christmas.",
    )


class DayForecast(BaseModel):
    """Forecast for a single day"""

    forecast_date: date = Field(description="Date of the forecast (YYYY-MM-DD)")
    point_forecast: float = Field(
        ge=0.0,
        description="Expected demand in units",
    )
    lower_90: float = Field(
        ge=0.0,
        description="Lower bound of the 90% prediction interval",
    )
    upper_90: float = Field(
        ge=0.0,
        description="Upper bound of the 90% prediction interval",
    )


class ForecastResponse(BaseModel):
    """Response body for POST /forecast endpoint
    Contains the full forecasthorizon with prediction intervals and metadata about the model that generated the forecast"""

    store_id: int
    start_date: date
    horizon_days: int
    forecasts: list[DayForecast] = Field(
        description="One forecast entry per day in the horizon"
    )
    model_version: str = Field(
        description="identifier for the model version that generated the forecast"
    )
    generated_at: str = Field(
        description="ISO 8601 timestamp when the forecast was generated"
    )


class HealthResponse(BaseModel):
    """Response body for GET /health endpoint
    Used for health checks and monitoring to confirm that the API is running and can respond to requests"""

    status: str
    version: str


class ReadyResponse(BaseModel):
    """Response body for GET /ready endpoint
    Used for readiness checks to confirm that the API is fully initialized and ready to handle forecast requests"""

    status: str
    model_loaded: bool
    detail: str | None = None
