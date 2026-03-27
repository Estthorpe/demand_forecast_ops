# src/models/validation.py
"""
Walk-forward validation for time series models.

Standard k-fold cross-validation is invalid for time series because
it shuffles data, allowing a model trained on future data to be
evaluated on past data — producing falsely optimistic metrics.

Walk-forward validation enforces temporal ordering:
- Training window always precedes evaluation window
- The training window expands with each fold (expanding window strategy)
- Each fold produces independent metrics
- Final report shows mean ± std across all folds — not a single number

Why mean ± std matters:
A model with SMAPE=14% on one fold but SMAPE=32% on another is
unreliable. Reporting only the mean hides this variance.
A senior engineer reviewing your work will ask for the distribution.
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger
from src.models.metrics import compute_all_metrics

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """
    Metrics and metadata from a single validation fold.

    Stored as a dataclass so it can be easily serialised to JSON
    for MLflow logging and the backtest report.
    """

    fold_number: int
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    smape: float
    mase: float
    coverage_90: float | None
    n_train_rows: int
    n_eval_rows: int


class ForecastModel(Protocol):
    """
    Protocol defining the interface any model must implement
    to be used with walk-forward validation.

    Using Protocol (structural subtyping) rather than inheritance
    means ARIMA, LightGBM, and any future model can be evaluated
    with the same validation code without any coupling.
    """

    def fit(self, train_df: pd.DataFrame) -> None:
        """Train on the given DataFrame."""
        ...

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts.

        Returns:
            Tuple of (point_forecasts, lower_bounds, upper_bounds).
            All arrays have length equal to horizon.
        """
        ...


def walk_forward_validate(
    df: pd.DataFrame,
    model: ForecastModel,
    n_folds: int = 5,
    eval_horizon_days: int = 14,
    min_train_days: int = 180,
) -> list[FoldResult]:
    """
    Run walk-forward expanding window validation.

    For each fold:
    1. Train on all data up to fold cutoff date
    2. Evaluate on the next eval_horizon_days days
    3. Advance cutoff by eval_horizon_days
    4. Repeat

    Args:
        df:                Full validated DataFrame with features.
        model:             A model implementing the ForecastModel protocol.
        n_folds:           Number of validation folds.
        eval_horizon_days: Days in each evaluation window.
        min_train_days:    Minimum training history required.

    Returns:
        List of FoldResult objects — one per fold.
    """
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    all_dates = sorted(df["date"].unique())
    total_days = len(all_dates)

    # Calculate the starting position for fold evaluation
    # We need min_train_days of history before the first evaluation
    eval_days_needed = n_folds * eval_horizon_days
    available_for_eval = total_days - min_train_days

    if available_for_eval < eval_days_needed:
        raise ValueError(
            f"Not enough data for {n_folds} folds of {eval_horizon_days} days. "
            f"Have {available_for_eval} days available after min_train_days={min_train_days}. "
            f"Need {eval_days_needed} days. "
            "Reduce n_folds or min_train_days."
        )

    results: list[FoldResult] = []

    # Starting cutoff: min_train_days before the last eval period
    first_eval_start_idx = total_days - eval_days_needed
    cutoff_idx = first_eval_start_idx

    for fold in range(n_folds):
        cutoff_date = all_dates[cutoff_idx]
        eval_end_idx = min(cutoff_idx + eval_horizon_days, total_days - 1)
        eval_end_date = all_dates[eval_end_idx]

        # Split data
        train_df = df[df["date"] <= cutoff_date].sort_values(["store", "date"]).copy()
        eval_df = df[(df["date"] > cutoff_date) & (df["date"] <= eval_end_date)]

        eval_df = eval_df.sort_values(["store", "date"]).copy()
        if len(train_df) == 0 or len(eval_df) == 0:
            logger.warning(
                "Fold {fold}: empty split — skipping",
                fold=fold + 1,
            )
            continue

        logger.info(
            "Fold {fold}/{total}: train={train_rows:,} rows "
            "({train_start}→{train_end}), "
            "eval={eval_rows:,} rows ({eval_start}→{eval_end})",
            fold=fold + 1,
            total=n_folds,
            train_rows=len(train_df),
            train_start=train_df["date"].min().date(),
            train_end=cutoff_date.date(),
            eval_rows=len(eval_df),
            eval_start=eval_df["date"].min().date(),
            eval_end=eval_end_date.date(),
        )

        # Train
        model.fit(train_df)

        # Predict
        point, lower, upper = model.predict(eval_df, horizon=eval_horizon_days)
        actual = eval_df["sales"].values

        # Align lengths — eval_df may have multiple stores
        min_len = min(len(actual), len(point))
        actual = actual[:min_len]
        point = point[:min_len]
        lower = lower[:min_len]
        upper = upper[:min_len]

        # Compute metrics
        fold_metrics = compute_all_metrics(actual, point, lower, upper)

        result = FoldResult(
            fold_number=fold + 1,
            train_start=train_df["date"].min().date(),
            train_end=cutoff_date.date(),
            eval_start=eval_df["date"].min().date(),
            eval_end=eval_end_date.date(),
            smape=fold_metrics["smape"],
            mase=fold_metrics["mase"],
            coverage_90=fold_metrics.get("coverage_90"),
            n_train_rows=len(train_df),
            n_eval_rows=len(eval_df),
        )

        results.append(result)
        logger.info(
            "Fold {fold} complete — SMAPE={smape:.2f}%, MASE={mase:.4f}",
            fold=fold + 1,
            smape=result.smape,
            mase=result.mase,
        )

        # Advance cutoff for next fold
        cutoff_idx += eval_horizon_days

    return results


def summarise_folds(results: list[FoldResult]) -> dict[str, float]:
    """
    Compute mean and std of each metric across all folds.

    This is what gets logged to MLflow as the final model metrics.
    Reporting mean ± std makes variance visible — a model that
    achieves good mean SMAPE but high std is unreliable in production.

    Returns:
        Dictionary with mean and std for each metric.
    """
    smape_values = [r.smape for r in results]
    mase_values = [r.mase for r in results]

    summary: dict[str, float] = {
        "smape_mean": float(np.mean(smape_values)),
        "smape_std": float(np.std(smape_values)),
        "mase_mean": float(np.mean(mase_values)),
        "mase_std": float(np.std(mase_values)),
    }

    coverage_values = [r.coverage_90 for r in results if r.coverage_90 is not None]
    if coverage_values:
        summary["coverage_90_mean"] = float(np.mean(coverage_values))

    return summary
