"""
Forecasting evaluation metrics for demand-forecast-ops.

Metrics implemented:
- SMAPE: Symmetric Mean Absolute Percentage Error
- MASE:  Mean Absolute Scaled Error
- Coverage: Fraction of actuals within prediction intervals
"""

import numpy as np


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Range: [0, 200%]. (the Llwer the  better).
    0% = perfect. 200% = maximally wrong.

    Handles zero actuals by adding a small epsilon to the denominator.
    Without epsilon, SMAPE is undefined when actual=0 AND predicted=0.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    epsilon = (
        1e-8  # epsilon prevents division by zero when both actual and predicted are 0
    )
    denominator = np.abs(actual) + np.abs(predicted) + epsilon

    return float(100.0 * np.mean(2.0 * np.abs(actual - predicted) / denominator))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonal_period: int = 7,
) -> float:
    """
    Mean Absolute Scaled Error.

    Compares model MAE against a naive seasonal baseline.
    MASE < 1.0: model beats naive baseline.
    MASE >= 1.0: model does not beat naive baseline.

    Args:
        actual:          Array of actual values (single time series).
        predicted:       Array of predicted values.
        seasonal_period: Naive baseline lag period.

    Returns:
        MASE value. Below 1.0 is better than naive.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    if len(actual) <= seasonal_period:
        return float("inf")

    model_mae = np.mean(np.abs(actual - predicted))
    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return float("inf")

    return float(model_mae / naive_mae)


def coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Prediction interval coverage

    Measures the fraction of actual values that fall within the predicted [lower, upper] interval
    """
    actual = np.array(actual, dtype=float)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    within = (actual >= lower) & (actual <= upper)
    return float(np.mean(within))


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all metrics in one call
    Returns a dictionary suitable for MLflow logging
    """
    metrics: dict[str, float] = {
        "smape": smape(actual, predicted),
        "mase": mase(actual, predicted),
    }

    if lower is not None and upper is not None:
        metrics["coverage_90"] = coverage(actual, lower, upper)

    return metrics
