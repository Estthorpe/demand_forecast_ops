"""
Calendar and holiday features for the Rossmann dataset
"""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based calendar features derivved from the date column
    These features give the model information about seasonality and cyclical patterns that raw date cannot convey directlt
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Basic calendar components ─────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # ── Binary flags ──────────────────────────────────────────────────
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # ── Cyclical encoding ─────────────────────────────────────────────
    # Why cyclical encoding:
    # Day 0 (Monday) and day 6 (Sunday) are adjacent in the week,
    # but numerically they are 6 apart. A linear model treats them
    # as maximally different. Sine/cosine encoding places them close
    # together in feature space — correctly representing the cycle.
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Holiday proximity ─────────────────────────────────────────────
    # Encode the state holiday categories as ordinal
    # '0'=no holiday, 'a'/'b'/'c'=different holiday types
    holiday_map = {"0": 0, "a": 1, "b": 2, "c": 3}
    if "state_holiday" in df.columns:
        df["state_holiday_encoded"] = (
            df["state_holiday"].map(holiday_map).fillna(0).astype(int)
        )

    return df
