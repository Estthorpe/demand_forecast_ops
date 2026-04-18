# src/monitoring/retraining_trigger.py
"""
Retraining trigger logic.

Combines signals from drift.py and forecast_tracker.py into a
single retraining decision.

Design rules (non-negotiable):
  - Never triggers retraining directly
  - Produces a TriggerReport with a boolean flag and full audit trail
  - All decisions are traceable to specific threshold breaches
  - Reports are append-only — never overwrite historical decisions
  - Consumed by src/agents/ which owns the retraining orchestration

Trigger conditions (ANY of these sets the should_retrain=True):
  1. Drift detected: max PSI >= 0.2 on any monitored feature
  2. Forecast quality critical: MASE >= 2.0
  3. Distribution failure: negative predictions or values above max
  4. Combined signal: PSI >= 0.1 AND MASE >= 1.5 (moderate drift + degradation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config.logging_config import get_logger
from src.monitoring.drift import PSI_MODERATE, PSI_NO_CHANGE, DriftReport
from src.monitoring.forecast_tracker import (
    MASE_CRITICAL_THRESHOLD,
    MASE_WARNING_THRESHOLD,
    ForecastHealthReport,
)

logger = get_logger(__name__)

# ── Trigger log path ────────────────────────────────────────────────────────
# Append-only log — historical decisions are never overwritten
DEFAULT_TRIGGER_LOG_PATH = Path("logs/retraining_triggers.jsonl")


@dataclass
class TriggerReason:
    """A single reason contributing to the retraining decision."""

    rule: str  # Rule name — e.g. "drift_detected"
    condition: str  # Human-readable condition — e.g. "PSI=0.23 >= 0.20"
    severity: str  # "warning" or "critical"
    value: float  # The metric value that triggered this reason


@dataclass
class TriggerReport:
    """
    Full retraining trigger report.

    This is the authoritative output of the monitoring pipeline.
    Consumed by src/agents/ — never triggers retraining directly.

    Fields:
        should_retrain: Primary boolean decision flag.
        reasons:        All rules that fired — full audit trail.
        warnings:       Non-critical signals that did not trigger retraining.
        drift_summary:  Key metrics from DriftReport.
        forecast_summary: Key metrics from ForecastHealthReport.
    """

    report_id: str
    computed_at: str
    should_retrain: bool
    reasons: list[TriggerReason]
    warnings: list[str]
    drift_summary: dict
    forecast_summary: dict
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "computed_at": self.computed_at,
            "should_retrain": self.should_retrain,
            "recommended_action": self.recommended_action,
            "reasons": [
                {
                    "rule": r.rule,
                    "condition": r.condition,
                    "severity": r.severity,
                    "value": round(r.value, 4),
                }
                for r in self.reasons
            ],
            "warnings": self.warnings,
            "drift_summary": self.drift_summary,
            "forecast_summary": self.forecast_summary,
        }


def _generate_report_id() -> str:
    """Generate a unique report ID based on timestamp."""
    return f"trigger_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _recommended_action(
    should_retrain: bool,
    reasons: list[TriggerReason],
) -> str:
    """
    Generate a human-readable recommended action string.

    Used in runbook and agent orchestration.
    """
    if not should_retrain:
        return "No action required. Continue monitoring."

    critical = [r for r in reasons if r.severity == "critical"]
    warnings = [r for r in reasons if r.severity == "warning"]

    if critical:
        rules = ", ".join(r.rule for r in critical)
        return (
            f"RETRAIN REQUIRED. Critical threshold breached: {rules}. "
            "Schedule retraining within 24 hours."
        )
    elif warnings:
        rules = ", ".join(r.rule for r in warnings)
        return (
            f"RETRAIN RECOMMENDED. Combined warning signals: {rules}. "
            "Schedule retraining within 72 hours."
        )

    return "RETRAIN FLAGGED. Review reasons and schedule retraining."


def evaluate_trigger(
    drift_report: DriftReport,
    forecast_report: ForecastHealthReport,
) -> TriggerReport:
    """
    Evaluate retraining trigger conditions from monitoring reports.

    Applies all trigger rules and produces a TriggerReport with
    full audit trail. Never triggers retraining directly.

    Args:
        drift_report:     Output of drift.compute_drift()
        forecast_report:  Output of forecast_tracker.track_forecast_health()

    Returns:
        TriggerReport with should_retrain flag and full reasoning.
    """
    reasons: list[TriggerReason] = []
    warnings: list[str] = []

    logger.info("Evaluating retraining trigger conditions...")

    # ── Rule 1: Significant drift detected ─────────────────────────────────
    if drift_report.max_psi >= PSI_MODERATE:
        reason = TriggerReason(
            rule="drift_detected",
            condition=f"max_PSI={drift_report.max_psi:.4f} >= {PSI_MODERATE}",
            severity="critical",
            value=drift_report.max_psi,
        )
        reasons.append(reason)
        logger.warning(f"Rule fired: {reason.rule} — {reason.condition}")

    # ── Rule 2: Forecast quality critical ──────────────────────────────────
    if (
        forecast_report.error_stats is not None
        and forecast_report.error_stats.mase >= MASE_CRITICAL_THRESHOLD
        and forecast_report.error_stats.mase != -1.0
    ):
        reason = TriggerReason(
            rule="forecast_quality_critical",
            condition=(
                f"MASE={forecast_report.error_stats.mase:.4f} "
                f">= {MASE_CRITICAL_THRESHOLD}"
            ),
            severity="critical",
            value=forecast_report.error_stats.mase,
        )
        reasons.append(reason)
        logger.warning(f"Rule fired: {reason.rule} — {reason.condition}")

    # ── Rule 3: Prediction distribution failure ─────────────────────────────
    if not forecast_report.distribution_stats.distribution_healthy:
        pct_neg = forecast_report.distribution_stats.pct_negative
        pct_high = forecast_report.distribution_stats.pct_above_max

        if pct_neg > 0:
            reason = TriggerReason(
                rule="negative_predictions",
                condition=f"{pct_neg:.1%} of predictions are negative",
                severity="critical",
                value=pct_neg,
            )
            reasons.append(reason)
            logger.warning(f"Rule fired: {reason.rule} — {reason.condition}")

        if pct_high > 0:
            reason = TriggerReason(
                rule="predictions_above_maximum",
                condition=f"{pct_high:.1%} of predictions exceed maximum",
                severity="critical",
                value=pct_high,
            )
            reasons.append(reason)
            logger.warning(f"Rule fired: {reason.rule} — {reason.condition}")

    # ── Rule 4: Combined moderate signal ───────────────────────────────────
    # Moderate drift + degraded accuracy together warrant retraining
    # even if neither individually crosses the critical threshold
    moderate_drift = PSI_NO_CHANGE <= drift_report.max_psi < PSI_MODERATE
    degraded_accuracy = (
        forecast_report.error_stats is not None
        and forecast_report.error_stats.mase >= MASE_WARNING_THRESHOLD
        and forecast_report.error_stats.mase != -1.0
    )

    if moderate_drift and degraded_accuracy and forecast_report.error is not None:
        reason = TriggerReason(
            rule="combined_moderate_signal",
            condition=(
                f"PSI={drift_report.max_psi:.4f} >= {PSI_NO_CHANGE} "
                f"AND MASE={forecast_report.error_stats.mase:.4f} "
                f">= {MASE_WARNING_THRESHOLD}"
            ),
            severity="warning",
            value=drift_report.max_psi,
        )
        reasons.append(reason)
        logger.warning(f"Rule fired: {reason.rule} — {reason.condition}")

    # ── Collect non-triggering warnings ────────────────────────────────────
    warnings.extend(forecast_report.warnings)

    if moderate_drift and not degraded_accuracy:
        warnings.append(
            f"Moderate drift detected (PSI={drift_report.max_psi:.4f}). "
            "Monitor closely."
        )

    # ── Final decision ──────────────────────────────────────────────────────
    should_retrain = len(reasons) > 0

    # Build summaries for report
    drift_summary = {
        "max_psi": round(drift_report.max_psi, 4),
        "drift_detected": drift_report.drift_detected,
        "drifted_features": drift_report.drifted_features,
        "monitored_features": drift_report.monitored_features,
    }

    forecast_summary: dict = {
        "distribution_healthy": (
            forecast_report.distribution_stats.distribution_healthy
        ),
        "prediction_count": forecast_report.distribution_stats.count,
        "mean_forecast": round(forecast_report.distribution_stats.mean, 2),
    }

    if forecast_report.error_stats is not None:
        forecast_summary.update(
            {
                "mase": round(forecast_report.error_stats.mase, 4),
                "smape": round(forecast_report.error_stats.smape, 4),
                "coverage_90": round(forecast_report.error_stats.coverage_90, 4),
                "mase_status": forecast_report.error_stats.mase_status,
            }
        )

    action = _recommended_action(should_retrain, reasons)

    report = TriggerReport(
        report_id=_generate_report_id(),
        computed_at=datetime.now(UTC).isoformat(),
        should_retrain=should_retrain,
        reasons=reasons,
        warnings=warnings,
        drift_summary=drift_summary,
        forecast_summary=forecast_summary,
        recommended_action=action,
    )

    logger.info(
        f"Trigger evaluation complete: should_retrain={should_retrain}, "
        f"rules_fired={len(reasons)}, action='{action}'"
    )

    return report


def append_trigger_log(
    report: TriggerReport,
    log_path: Path = DEFAULT_TRIGGER_LOG_PATH,
) -> None:
    """
    Append trigger report to the audit log.

    Uses JSONL format — one JSON object per line.
    Append-only — historical decisions are never overwritten.
    This log is the audit trail for all retraining decisions.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(json.dumps(report.to_dict()) + "\n")

    logger.info(f"Trigger report {report.report_id} appended to {log_path}")


def load_trigger_history(
    log_path: Path = DEFAULT_TRIGGER_LOG_PATH,
) -> list[dict]:
    """
    Load full trigger history from audit log.

    Returns list of all historical trigger reports in
    chronological order.
    """
    if not log_path.exists():
        logger.info("No trigger history found — returning empty list")
        return []

    reports = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupt trigger log entry: {e}")
                    continue

    logger.info(f"Loaded {len(reports)} trigger reports from {log_path}")
    return reports


def get_latest_trigger(
    log_path: Path = DEFAULT_TRIGGER_LOG_PATH,
) -> dict | None:
    """Return the most recent trigger report, or None if log is empty."""
    history = load_trigger_history(log_path)
    return history[-1] if history else None
