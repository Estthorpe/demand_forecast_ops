# src/features/pipeline.py
"""
Assembled feature pipeline for demand-forecast-ops.

This is the SINGLE entry point for all feature computation.
Both the training script and the serving endpoint call build_features().

If you need to change how features are computed, change it here.
Never compute features differently in training vs serving — that is
how training-serving skew happens.
"""

import pandas as pd

from src.config.logging_config import get_logger
from src.features.calendar import add_calendar_features
from src.features.lag_features import add_lag_features, add_rolling_features

logger = get_logger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature pipeline to a validated DataFrame.

    Order matters:
    1. Calendar features — derived from date, no dependency on sales
    2. Lag features — derived from past sales values
    3. Rolling features — derived from past sales windows

    Lag and rolling features shift values by 1 before computing,
    ensuring no information from the current day leaks into the features.
    """
    logger.info(f"Building features for {len(df)} rows")

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop rows with NaN lag features — these are the first
    # LAG_PERIODS rows per store where we do not have enough history.
    # Training on NaN values causes silent model degradation.
    rows_before = len(df)
    df = df.dropna(subset=[c for c in df.columns if "lag" in c])
    rows_dropped = rows_before - len(df)

    if rows_dropped > 0:
        logger.info(
            f"Dropped {rows_dropped} rows with NaN lag features "
            "(expected — first 28 days per store)"
        )

    if "store_type" in df.columns:
        store_type_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        df["store_type"] = df["store_type"].map(store_type_map).fillna(-1).astype(int)

    if "assortment" in df.columns:
        assortment_map = {"a": 0, "b": 1, "c": 2}
        df["assortment"] = df["assortment"].map(assortment_map).fillna(-1).astype(int)

    logger.info(
        f"Feature pipeline complete: {len(df)} rows, {len(df.columns)} features"
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns used for model training.

    Excludes: target variable, identifiers, and raw date.
    This list is what gets passed to the model's fit() and predict().
    Keeping it as a function — not a hardcoded list — means it
    automatically updates when new features are added to build_features().
    """
    exclude = {
        "sales",  # target variable
        "customers",  # not available at prediction time
        "store",  # identifier — not a feature
        "date",  # raw date — encoded features used instead
        "state_holiday",  # raw string — encoded version used instead
    }
    return [col for col in df.columns if col not in exclude]
