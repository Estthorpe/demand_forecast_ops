"""
Unit tests for metrics and feature pipeline
"""

import numpy as np
import pytest

from src.models.metrics import coverage, mase, smape


def test_smape_perfect_forecast() -> None:
    """Perfect forecast should produce SMAPE of 0"""
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([100.0, 200.0, 300.0])
    assert smape(actual, predicted) == pytest.approx(0.0, abs=1e-6)


def test_smape_handles_zeros() -> None:
    """SMAPE must not raise on zero values — epsilon prevents division by zero."""
    actual = np.array([0.0, 100.0])
    predicted = np.array([0.0, 110.0])
    result = smape(actual, predicted)
    assert np.isfinite(result)
    assert result >= 0.0


def test_mase_equals_one_for_naive_baseline() -> None:
    """
    A seasonal naive baseline should achieve MASE close to 1.0
    when evaluated against itself on a single clean time series.

    Uses a long repeating pattern with no boundary crossings
    so the boundary filter in mase() does not remove valid errors.
    """
    # Long repeating weekly pattern — 10 full weeks
    pattern = np.array([100.0, 120.0, 90.0, 110.0, 130.0, 80.0, 70.0])
    actual = np.tile(pattern, 10)  # 70 values, clean single series

    # Naive prediction: same day last week (shift by 7)
    # For positions 0-6, use the actual value (no lag available)
    predicted = np.concatenate([actual[:7], actual[:-7]])

    result = mase(actual, predicted)

    # MASE should be close to 1.0 for a naive predictor on a clean series
    # Allow tolerance of 0.2 to account for edge effects at boundaries
    assert result < 1.2, f"Expected MASE close to 1.0, got {result}"
    assert result > 0.8, f"Expected MASE close to 1.0, got {result}"


def test_coverage_perfect_intervals() -> None:
    """Intervals that contain all actuals should give coverage of 1.0."""
    actual = np.array([100.0, 200.0, 300.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1000.0, 1000.0, 1000.0])
    assert coverage(actual, lower, upper) == pytest.approx(1.0)


def test_coverage_no_containment() -> None:
    """Intervals that contain no actuals should give coverage of 0.0."""
    actual = np.array([500.0, 600.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])
    assert coverage(actual, lower, upper) == pytest.approx(0.0)
