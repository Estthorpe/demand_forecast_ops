"""
Lag and rolling window features for time series forecasting.

Critical constraint: these features must be computed with strict
temporal ordering. Computing rolling statistics without sorting by
date first causes future data leakage — the model sees tomorrow's
sales when predicting today.

All functions sort by date before computing any window operation.
"""

import pandas as pd

# Lag periods chosen:
# - t-1:  yesterday (immediate trend)
# - t-7:  same day last week (weekly seasonality)
# - t-14: two weeks ago (bi-weekly patterns)
# - t-28: four weeks ago (monthly seasonality)
LAG_PERIODS = [1, 7, 14, 28]

# Rolling windows for statistics
# 7-day and 28-day capture weekly and monthly stability
ROLLING_WINDOWS = [7, 28]


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "store",
    target_col: str = "sales",
    lag_periods: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lag features for the target variable, grouped by store.

    Grouping by store is critical: lag-7 for store 1 should reference
    store 1's sales 7 days ago, not any store's sales 7 days ago.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])

    periods = lag_periods or LAG_PERIODS

    for lag in periods:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "store",
    target_col: str = "sales",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.

    Rolling statistics capture the local trend and volatility around
    each prediction point. A store with rising 7-day rolling mean
    is trending upward — a useful signal for the model.

    The min_periods=1 argument ensures we get values even at the
    start of the time series, rather than NaN for the first window-1 rows.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])

    window_sizes = windows or ROLLING_WINDOWS

    for window in window_sizes:
        mean_col = f"{target_col}_rolling_mean_{window}"
        std_col = f"{target_col}_rolling_std_{window}"

        df[mean_col] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[std_col] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

    return df
