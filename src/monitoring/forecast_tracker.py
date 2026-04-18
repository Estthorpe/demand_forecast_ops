# src/monitoring/forecast_tracker.py
"""
Forecast error and prediction distribution tracker.

Reads inference logs produced by src/serving/ and tracks:
  1. Prediction distribution — catches unrealistic forecast values
  2. Forecast error metrics — when actuals become available
  3. Interval coverage — checks if 90% intervals actually contain actuals

This module never triggers retraining directly.
It produces a ForecastHealthReport consumed by retraining_trigger.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.config.logging_config import get_logger

logger = get_logger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────
# Prediction distribution thresholds
MAX_ALLOWED_FORECAST = 100_000.0  # Rossmann max realistic daily sales
MIN_ALLOWED_FORECAST = 0.0  # Sales cannot be negative

# Error thresholds (when actuals available)
MASE_WARNING_THRESHOLD = 1.5  # MASE > 1.5 → warning
MASE_CRITICAL_THRESHOLD = 2.0  # MASE > 2.0 → trigger retraining flag

# Coverage threshold for 90% prediction intervals
COVERAGE_WARNING_THRESHOLD = 0.75  # Expected ~0.90, warn below 0.75

# Minimum number of predictions before error metrics are meaningful
MIN_PREDICTIONS_FOR_EVAL = 10


@dataclass
class PredictionDistributionStats:
    """Statistics on raw prediction values."""

    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    pct_10: float
    pct_50: float
    pct_90: float
    pct_negative: float  # Should always be 0.0
    pct_above_max: float  # % predictions above MAX_ALLOWED_FORECAST
    distribution_healthy: bool


@dataclass
class ForecastErrorStats:
    """Error metrics computed when actuals are available."""

    count: int
    mase: float
    smape: float
    coverage_90: float  # Actual coverage of 90% intervals
    mase_status: str  # "healthy", "warning", "critical"
    coverage_status: str  # "healthy", "warning"


@dataclass
class ForecastHealthReport:
    """
    Full forecast health report.

    Consumed by retraining_trigger.py — never triggers
    retraining directly.
    """

    computed_at: str
    period: str
    distribution_stats: PredictionDistributionStats
    error_stats: ForecastErrorStats | None
    retraining_flag: bool  # True if any critical threshold breached
    warnings: list[str]
    mode: str  # "distribution_only" or "full_evaluation"

    def to_dict(self) -> dict:
        result = {
            "computed_at": self.computed_at,
            "period": self.period,
            "mode": self.mode,
            "retraining_flag": self.retraining_flag,
            "warnings": self.warnings,
            "distribution_stats": {
                "count": self.distribution_stats.count,
                "mean": round(self.distribution_stats.mean, 2),
                "std": round(self.distribution_stats.std, 2),
                "min": round(self.distribution_stats.min_val, 2),
                "max": round(self.distribution_stats.max_val, 2),
                "p10": round(self.distribution_stats.pct_10, 2),
                "p50": round(self.distribution_stats.pct_50, 2),
                "p90": round(self.distribution_stats.pct_90, 2),
                "pct_negative": round(self.distribution_stats.pct_negative, 4),
                "pct_above_max": round(self.distribution_stats.pct_above_max, 4),
                "distribution_healthy": self.distribution_stats.distribution_healthy,
            },
        }

        if self.error_stats:
            result["error_stats"] = {
                "count": self.error_stats.count,
                "mase": round(self.error_stats.mase, 4),
                "smape": round(self.error_stats.smape, 4),
                "coverage_90": round(self.error_stats.coverage_90, 4),
                "mase_status": self.error_stats.mase_status,
                "coverage_status": self.error_stats.coverage_status,
            }

        return result


def _compute_distribution_stats(
    predictions: np.ndarray,
) -> PredictionDistributionStats:
    """
    Compute distribution statistics on raw prediction values.

    Catches model collapse (all predictions near zero),
    explosion (unrealistically high values), and negative
    predictions which should never occur for sales data.
    """
    count = len(predictions)

    if count == 0:
        logger.warning("Empty predictions array — returning zero stats")
        return PredictionDistributionStats(
            count=0,
            mean=0.0,
            std=0.0,
            min_val=0.0,
            max_val=0.0,
            pct_10=0.0,
            pct_50=0.0,
            pct_90=0.0,
            pct_negative=0.0,
            pct_above_max=0.0,
            distribution_healthy=False,
        )

    pct_negative = float(np.mean(predictions < MIN_ALLOWED_FORECAST))
    pct_above_max = float(np.mean(predictions > MAX_ALLOWED_FORECAST))
    distribution_healthy = (pct_negative == 0.0) and (pct_above_max == 0.0)

    return PredictionDistributionStats(
        count=count,
        mean=float(np.mean(predictions)),
        std=float(np.std(predictions)),
        min_val=float(np.min(predictions)),
        max_val=float(np.max(predictions)),
        pct_10=float(np.percentile(predictions, 10)),
        pct_50=float(np.percentile(predictions, 50)),
        pct_90=float(np.percentile(predictions, 90)),
        pct_negative=pct_negative,
        pct_above_max=pct_above_max,
        distribution_healthy=distribution_healthy,
    )


def _compute_mase(
    actuals: np.ndarray,
    predictions: np.ndarray,
    seasonal_period: int = 7,
) -> float:
    """
    Compute Mean Absolute Scaled Error.

    Uses seasonal naive baseline (period=7 for weekly seasonality)
    as the scaling denominator — consistent with Phase 3 evaluation.
    """
    if len(actuals) < seasonal_period + 1:
        logger.warning(f"Insufficient data for MASE (need >{seasonal_period} points)")
        return float("nan")

    mae = np.mean(np.abs(actuals - predictions))

    # Seasonal naive baseline MAE
    naive_mae = np.mean(np.abs(actuals[seasonal_period:] - actuals[:-seasonal_period]))

    if naive_mae == 0:
        return float("nan")

    return float(mae / naive_mae)


def _compute_smape(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(actuals) + np.abs(predictions)) / 2
    # Avoid division by zero
    mask = denominator > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actuals[mask] - predictions[mask]) / denominator[mask]))


def _compute_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Compute actual coverage of prediction intervals.

    For 90% intervals, expected coverage is ~0.90.
    Significant deviation indicates interval miscalibration.
    """
    within_interval = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(within_interval))


def track_forecast_health(
    predictions: np.ndarray,
    period: str = "current",
    actuals: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> ForecastHealthReport:
    """
    Compute forecast health report from prediction arrays.

    Args:
        predictions:   Point forecast values from model.
        period:        Label for the monitoring period.
        actuals:       Actual sales values if available (retrospective).
        lower_bounds:  Lower 90% interval bounds (for coverage check).
        upper_bounds:  Upper 90% interval bounds (for coverage check).

    Returns:
        ForecastHealthReport with distribution stats and optional
        error stats. retraining_flag=True if critical threshold breached.
    """
    logger.info(
        f"Tracking forecast health: {len(predictions)} predictions, "
        f"actuals_available={actuals is not None}"
    )

    warnings = []
    retraining_flag = False

    # Always compute distribution stats
    dist_stats = _compute_distribution_stats(predictions)

    if not dist_stats.distribution_healthy:
        if dist_stats.pct_negative > 0:
            msg = f"{dist_stats.pct_negative:.1%} of predictions are negative"
            warnings.append(msg)
            logger.warning(msg)
            retraining_flag = True

        if dist_stats.pct_above_max > 0:
            msg = (
                f"{dist_stats.pct_above_max:.1%} of predictions "
                f"exceed maximum ({MAX_ALLOWED_FORECAST:,.0f})"
            )
            warnings.append(msg)
            logger.warning(msg)
            retraining_flag = True

    # Error stats only when actuals are available
    error_stats = None
    mode = "distribution_only"

    if actuals is not None and len(actuals) >= MIN_PREDICTIONS_FOR_EVAL:
        mode = "full_evaluation"

        mase = _compute_mase(actuals, predictions)
        smape = _compute_smape(actuals, predictions)

        # MASE status
        if np.isnan(mase):
            mase_status = "insufficient_data"
        elif mase >= MASE_CRITICAL_THRESHOLD:
            mase_status = "critical"
            retraining_flag = True
            warnings.append(f"MASE={mase:.3f} exceeds critical threshold")
        elif mase >= MASE_WARNING_THRESHOLD:
            mase_status = "warning"
            warnings.append(f"MASE={mase:.3f} exceeds warning threshold")
        else:
            mase_status = "healthy"

        # Coverage stats
        coverage_90 = float("nan")
        coverage_status = "unavailable"

        if lower_bounds is not None and upper_bounds is not None:
            coverage_90 = _compute_coverage(actuals, lower_bounds, upper_bounds)
            if coverage_90 < COVERAGE_WARNING_THRESHOLD:
                coverage_status = "warning"
                warnings.append(
                    f"Interval coverage={coverage_90:.1%} below "
                    f"threshold ({COVERAGE_WARNING_THRESHOLD:.0%})"
                )
            else:
                coverage_status = "healthy"

        error_stats = ForecastErrorStats(
            count=len(actuals),
            mase=mase if not np.isnan(mase) else -1.0,
            smape=smape if not np.isnan(smape) else -1.0,
            coverage_90=coverage_90 if not np.isnan(coverage_90) else -1.0,
            mase_status=mase_status,
            coverage_status=coverage_status,
        )

        logger.info(
            f"Error stats: MASE={mase:.4f} ({mase_status}), "
            f"SMAPE={smape:.4f}, coverage={coverage_90:.4f}"
        )

    report = ForecastHealthReport(
        computed_at=datetime.now(UTC).isoformat(),
        period=period,
        distribution_stats=dist_stats,
        error_stats=error_stats,
        retraining_flag=retraining_flag,
        warnings=warnings,
        mode=mode,
    )

    logger.info(
        f"Forecast health: retraining_flag={retraining_flag}, "
        f"warnings={len(warnings)}, mode={mode}"
    )

    return report


def save_forecast_health_report(
    report: ForecastHealthReport,
    output_path: Path,
) -> None:
    """Save forecast health report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Forecast health report saved to {output_path}")
