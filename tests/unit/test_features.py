"""
Unit tests for metrics and feature pipeline
"""

import numpy as np
import pytest

from src.models.metrics import coverage, smape


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
