# src/monitoring/drift.py
"""
Population Stability Index (PSI) drift detection.

Reads inference logs produced by src/serving/ and computes PSI
against the training reference distribution.

PSI thresholds (industry standard):
  PSI < 0.1  → No significant change
  PSI < 0.2  → Moderate change — monitor closely
  PSI >= 0.2 → Significant drift — trigger retraining flag

This module never triggers retraining directly.
It produces a DriftReport and a boolean flag consumed by
retraining_trigger.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)

# ── PSI thresholds ─────────────────────────────────────────────────────────
PSI_NO_CHANGE = 0.1
PSI_MODERATE = 0.2  # >= this value triggers retraining flag

# ── Features monitored for drift ───────────────────────────────────────────
# Calendar features only — these are available at inference time (T-012)
MONITORED_FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "promo",
]

# Number of bins for PSI computation
N_BINS = 10


@dataclass
class FeatureDriftResult:
    """PSI result for a single feature."""

    feature: str
    psi: float
    status: str  # "stable", "moderate", "drifted"
    reference_count: int
    current_count: int


@dataclass
class DriftReport:
    """
    Full drift report produced by compute_drift().

    Consumed by retraining_trigger.py — never triggers
    retraining directly.
    """

    computed_at: str
    reference_period: str
    current_period: str
    feature_results: list[FeatureDriftResult]
    max_psi: float
    drift_detected: bool  # True if any feature PSI >= PSI_MODERATE
    monitored_features: int
    drifted_features: int

    def to_dict(self) -> dict:
        return {
            "computed_at": self.computed_at,
            "reference_period": self.reference_period,
            "current_period": self.current_period,
            "max_psi": round(self.max_psi, 4),
            "drift_detected": self.drift_detected,
            "monitored_features": self.monitored_features,
            "drifted_features": self.drifted_features,
            "feature_results": [
                {
                    "feature": r.feature,
                    "psi": round(r.psi, 4),
                    "status": r.status,
                    "reference_count": r.reference_count,
                    "current_count": r.current_count,
                }
                for r in self.feature_results
            ],
        }


def _psi_status(psi: float) -> str:
    """Classify PSI value into human-readable status."""
    if psi < PSI_NO_CHANGE:
        return "stable"
    elif psi < PSI_MODERATE:
        return "moderate"
    else:
        return "drifted"


def _compute_psi_for_feature(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = N_BINS,
) -> float:
    """
    Compute PSI between reference and current distributions.

    Uses reference distribution to define bin edges — this
    ensures consistent binning across time periods.

    Args:
        reference: Feature values from training/reference period.
        current:   Feature values from current inference period.
        n_bins:    Number of histogram bins.

    Returns:
        PSI value (float). Returns 0.0 if computation fails.
    """
    try:
        # Define bins from reference distribution
        min_val = reference.min()
        max_val = reference.max()

        # Handle constant features (no variance)
        if min_val == max_val:
            logger.warning("Constant feature detected — PSI set to 0.0")
            return 0.0

        bins = np.linspace(min_val, max_val, n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Compute proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        ref_props = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
        cur_props = (cur_counts + epsilon) / (len(current) + epsilon * n_bins)

        # PSI formula
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi)

    except Exception as e:
        logger.error(f"PSI computation failed: {e}")
        return 0.0


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_period: str = "training",
    current_period: str = "current",
) -> DriftReport:
    """
    Compute PSI drift between reference and current DataFrames.

    Args:
        reference_df:     Training or reference feature DataFrame.
        current_df:       Current inference feature DataFrame.
        reference_period: Label for reference period (for report).
        current_period:   Label for current period (for report).

    Returns:
        DriftReport with per-feature PSI and overall drift flag.
    """
    logger.info(
        f"Computing drift: reference={len(reference_df)} rows, "
        f"current={len(current_df)} rows"
    )

    feature_results = []

    for feature in MONITORED_FEATURES:
        # Skip features not present in both DataFrames
        if feature not in reference_df.columns:
            logger.warning(f"Feature '{feature}' missing from reference — skipping")
            continue
        if feature not in current_df.columns:
            logger.warning(f"Feature '{feature}' missing from current — skipping")
            continue

        ref_series = reference_df[feature].dropna()
        cur_series = current_df[feature].dropna()

        if len(ref_series) == 0 or len(cur_series) == 0:
            logger.warning(f"Empty series for feature '{feature}' — skipping")
            continue

        psi = _compute_psi_for_feature(ref_series, cur_series)
        status = _psi_status(psi)

        feature_results.append(
            FeatureDriftResult(
                feature=feature,
                psi=psi,
                status=status,
                reference_count=len(ref_series),
                current_count=len(cur_series),
            )
        )

        logger.info(f"  {feature}: PSI={psi:.4f} ({status})")

    # Compute summary statistics
    max_psi = max((r.psi for r in feature_results), default=0.0)
    drift_detected = max_psi >= PSI_MODERATE
    drifted_features = sum(1 for r in feature_results if r.status == "drifted")

    report = DriftReport(
        computed_at=datetime.now(UTC).isoformat(),
        reference_period=reference_period,
        current_period=current_period,
        feature_results=feature_results,
        max_psi=max_psi,
        drift_detected=drift_detected,
        monitored_features=len(feature_results),
        drifted_features=drifted_features,
    )

    logger.info(
        f"Drift report: max_psi={max_psi:.4f}, "
        f"drift_detected={drift_detected}, "
        f"drifted_features={drifted_features}/{len(feature_results)}"
    )

    return report


def save_drift_report(report: DriftReport, output_path: Path) -> None:
    """Save drift report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Drift report saved to {output_path}")


def load_reference_distribution(reference_path: Path) -> pd.DataFrame:
    """
    Load reference feature distribution from parquet or CSV.

    The reference distribution is computed once from training data
    and saved during the training pipeline. It is the baseline
    against which all future inference data is compared.
    """
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference distribution not found at {reference_path}. "
            "Run scripts/compute_reference_distribution.py first."
        )

    if reference_path.suffix == ".parquet":
        return pd.read_parquet(reference_path)
    elif reference_path.suffix == ".csv":
        return pd.read_csv(reference_path)
    else:
        raise ValueError(f"Unsupported format: {reference_path.suffix}")
