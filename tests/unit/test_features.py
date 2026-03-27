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


def test_mase_better_model_scores_lower() -> None:
    """
    A model with smaller errors should score lower MASE than
    a model with larger errors.
    """
    actual = np.array(
        [
            100.0,
            120.0,
            90.0,
            110.0,
            130.0,
            80.0,
            70.0,
            100.0,
            120.0,
            90.0,
            110.0,
            130.0,
            80.0,
            70.0,
            100.0,
            120.0,
            90.0,
            110.0,
            130.0,
            80.0,
            70.0,
        ]
    )

    good_predictions = actual + 5.0  # small constant error
    bad_predictions = actual + 50.0  # large constant error

    good_mase = mase(actual, good_predictions)
    bad_mase = mase(actual, bad_predictions)

    assert np.isfinite(good_mase), "Good model MASE should be finite"
    assert np.isfinite(bad_mase), "Bad model MASE should be finite"
    assert (
        good_mase < bad_mase
    ), f"Good model MASE {good_mase} should be less than bad model MASE {bad_mase}"


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
