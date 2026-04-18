# tests/unit/test_monitoring.py
"""
Unit tests for Phase 5 monitoring modules.

Tests cover:
  - PSI computation correctness
  - Drift report structure and thresholds
  - Forecast health report modes
  - Retraining trigger rule evaluation
  - Audit log append behaviour
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    PSI_MODERATE,
    PSI_NO_CHANGE,
    DriftReport,
    _compute_psi_for_feature,
    _psi_status,
    compute_drift,
    save_drift_report,
)
from src.monitoring.forecast_tracker import (
    ForecastHealthReport,
    save_forecast_health_report,
    track_forecast_health,
)
from src.monitoring.retraining_trigger import (
    TriggerReport,
    append_trigger_log,
    evaluate_trigger,
    load_trigger_history,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Reference distribution — clean retail sales pattern."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "day_of_week": np.random.randint(0, 7, n),
            "month": np.random.randint(1, 13, n),
            "day_of_month": np.random.randint(1, 29, n),
            "is_weekend": np.random.randint(0, 2, n),
            "promo": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def current_df_stable(reference_df) -> pd.DataFrame:
    """Current distribution — same as reference (no drift)."""
    np.random.seed(99)
    n = 100
    return pd.DataFrame(
        {
            "day_of_week": np.random.randint(0, 7, n),
            "month": np.random.randint(1, 13, n),
            "day_of_month": np.random.randint(1, 29, n),
            "is_weekend": np.random.randint(0, 2, n),
            "promo": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def current_df_drifted() -> pd.DataFrame:
    """Current distribution — heavily skewed (drift scenario)."""
    n = 100
    return pd.DataFrame(
        {
            "day_of_week": np.zeros(n, dtype=int),  # All Monday
            "month": np.ones(n, dtype=int) * 12,  # All December
            "day_of_month": np.ones(n, dtype=int) * 25,  # All 25th
            "is_weekend": np.zeros(n, dtype=int),  # All weekday
            "promo": np.ones(n, dtype=int),  # All promo
        }
    )


@pytest.fixture
def healthy_predictions() -> np.ndarray:
    """Realistic sales predictions."""
    np.random.seed(42)
    return np.random.uniform(500, 8000, 100)


@pytest.fixture
def stable_drift_report(reference_df, current_df_stable) -> DriftReport:
    return compute_drift(reference_df, current_df_stable)


@pytest.fixture
def drifted_drift_report(reference_df, current_df_drifted) -> DriftReport:
    return compute_drift(reference_df, current_df_drifted)


@pytest.fixture
def healthy_forecast_report(healthy_predictions) -> ForecastHealthReport:
    return track_forecast_health(healthy_predictions, period="test")


# ── PSI unit tests ───────────────────────────────────────────────────────────


class TestPSIComputation:
    def test_identical_distributions_give_near_zero_psi(self):
        """Identical distributions should produce PSI close to zero."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = _compute_psi_for_feature(
            pd.Series(data[:500]),
            pd.Series(data[500:]),
        )
        assert psi < PSI_NO_CHANGE, f"Expected PSI < {PSI_NO_CHANGE}, got {psi}"

    def test_different_distributions_give_high_psi(self):
        """Distributions with no overlap should produce high PSI."""
        np.random.seed(0)
        reference = pd.Series(np.random.normal(0, 0.1, 500))
        current = pd.Series(np.random.normal(100, 0.1, 500))
        psi = _compute_psi_for_feature(reference, current)
        assert psi >= PSI_MODERATE, f"Expected PSI >= {PSI_MODERATE}, got {psi}"

    def test_constant_feature_returns_zero(self):
        """Constant feature (no variance) should return PSI=0.0."""
        constant = pd.Series(np.ones(100) * 5.0)
        psi = _compute_psi_for_feature(constant, constant.copy())
        assert psi == 0.0

    def test_psi_status_classification(self):
        assert _psi_status(0.05) == "stable"
        assert _psi_status(0.12) == "moderate"
        assert _psi_status(0.25) == "drifted"
        assert _psi_status(PSI_NO_CHANGE) == "moderate"
        assert _psi_status(PSI_MODERATE) == "drifted"


# ── Drift report tests ───────────────────────────────────────────────────────


class TestDriftReport:
    def test_stable_drift_report_structure(self, stable_drift_report):
        """Stable report has correct structure and no drift detected."""
        assert isinstance(stable_drift_report, DriftReport)
        assert stable_drift_report.monitored_features > 0
        assert stable_drift_report.max_psi >= 0.0
        assert isinstance(stable_drift_report.drift_detected, bool)

    def test_stable_distributions_no_drift(self, stable_drift_report):
        """Similar distributions should not trigger drift flag."""
        assert not stable_drift_report.drift_detected

    def test_drifted_distributions_triggers_flag(self, drifted_drift_report):
        """Heavily skewed distributions should trigger drift flag."""
        assert drifted_drift_report.drift_detected
        assert drifted_drift_report.max_psi >= PSI_MODERATE

    def test_drift_report_serialisable(self, stable_drift_report):
        """DriftReport must serialise to dict without errors."""
        d = stable_drift_report.to_dict()
        assert "max_psi" in d
        assert "drift_detected" in d
        assert "feature_results" in d
        assert isinstance(d["feature_results"], list)

    def test_save_drift_report(self, stable_drift_report, tmp_path):
        """Drift report saves to JSON correctly."""
        output = tmp_path / "drift_report.json"
        save_drift_report(stable_drift_report, output)
        assert output.exists()
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["drift_detected"] == stable_drift_report.drift_detected


# ── Forecast tracker tests ───────────────────────────────────────────────────


class TestForecastTracker:
    def test_healthy_predictions_no_flag(self, healthy_predictions):
        """Realistic predictions should not raise retraining flag."""
        report = track_forecast_health(healthy_predictions, period="test")
        assert not report.retraining_flag
        assert report.distribution_stats.distribution_healthy

    def test_negative_predictions_raises_flag(self):
        """Negative predictions must raise retraining flag."""
        bad_predictions = np.array([-100.0, -200.0, -50.0] * 10)
        report = track_forecast_health(bad_predictions, period="test")
        assert report.retraining_flag
        assert report.distribution_stats.pct_negative > 0

    def test_excessive_predictions_raises_flag(self):
        """Predictions above maximum must raise retraining flag."""
        bad_predictions = np.ones(50) * 200_000.0
        report = track_forecast_health(bad_predictions, period="test")
        assert report.retraining_flag
        assert report.distribution_stats.pct_above_max > 0

    def test_distribution_only_mode_without_actuals(self, healthy_predictions):
        """Without actuals, mode should be distribution_only."""
        report = track_forecast_health(healthy_predictions)
        assert report.mode == "distribution_only"
        assert report.error_stats is None

    def test_full_evaluation_mode_with_actuals(self, healthy_predictions):
        """With actuals, mode should be full_evaluation."""
        actuals = healthy_predictions * np.random.uniform(
            0.9, 1.1, len(healthy_predictions)
        )
        report = track_forecast_health(
            healthy_predictions,
            actuals=actuals,
        )
        assert report.mode == "full_evaluation"
        assert report.error_stats is not None

    def test_critical_mase_raises_flag(self):
        """MASE above critical threshold must raise retraining flag."""
        np.random.seed(42)
        n = 50
        actuals = np.random.uniform(5000, 10000, n)
        # Predictions near zero — guaranteed critical MASE
        predictions = np.ones(n) * 10.0
        report = track_forecast_health(
            predictions,
            actuals=actuals,
        )
        assert report.retraining_flag
        assert report.error_stats.mase_status == "critical"

    def test_report_serialisable(self, healthy_forecast_report):
        """ForecastHealthReport must serialise without errors."""
        d = healthy_forecast_report.to_dict()
        assert "distribution_stats" in d
        assert "retraining_flag" in d
        assert "mode" in d

    def test_save_forecast_report(self, healthy_forecast_report, tmp_path):
        """Forecast health report saves to JSON correctly."""
        output = tmp_path / "forecast_report.json"
        save_forecast_health_report(healthy_forecast_report, output)
        assert output.exists()


# ── Trigger evaluation tests ─────────────────────────────────────────────────


class TestRetrainingTrigger:
    def test_no_trigger_when_stable(
        self,
        stable_drift_report,
        healthy_forecast_report,
    ):
        """Stable drift + healthy forecasts should not trigger retraining."""
        report = evaluate_trigger(stable_drift_report, healthy_forecast_report)
        assert isinstance(report, TriggerReport)
        assert not report.should_retrain
        assert len(report.reasons) == 0

    def test_trigger_on_significant_drift(
        self,
        drifted_drift_report,
        healthy_forecast_report,
    ):
        """Significant drift alone should trigger retraining."""
        report = evaluate_trigger(drifted_drift_report, healthy_forecast_report)
        assert report.should_retrain
        rule_names = [r.rule for r in report.reasons]
        assert "drift_detected" in rule_names

    def test_trigger_report_has_recommended_action(
        self,
        drifted_drift_report,
        healthy_forecast_report,
    ):
        """Trigger report must always include a recommended action."""
        report = evaluate_trigger(drifted_drift_report, healthy_forecast_report)
        assert report.recommended_action
        assert len(report.recommended_action) > 0

    def test_trigger_report_serialisable(
        self,
        stable_drift_report,
        healthy_forecast_report,
    ):
        """TriggerReport must serialise to dict without errors."""
        report = evaluate_trigger(stable_drift_report, healthy_forecast_report)
        d = report.to_dict()
        assert "should_retrain" in d
        assert "reasons" in d
        assert "recommended_action" in d
        assert "report_id" in d

    def test_audit_log_append(
        self,
        stable_drift_report,
        healthy_forecast_report,
        tmp_path,
    ):
        """Trigger log is append-only — multiple reports stack correctly."""
        import time

        log_path = tmp_path / "triggers.jsonl"

        report1 = evaluate_trigger(stable_drift_report, healthy_forecast_report)
        append_trigger_log(report1, log_path)

        time.sleep(1)  # Ensure different timestamp-based report_id

        report2 = evaluate_trigger(stable_drift_report, healthy_forecast_report)
        append_trigger_log(report2, log_path)

        history = load_trigger_history(log_path)
        assert len(history) == 2
        assert history[0]["report_id"] != history[1]["report_id"]

    def test_audit_log_empty_when_no_history(self, tmp_path):
        """Loading non-existent log returns empty list."""
        log_path = tmp_path / "nonexistent.jsonl"
        history = load_trigger_history(log_path)
        assert history == []
