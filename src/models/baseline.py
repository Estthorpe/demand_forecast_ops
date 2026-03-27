"""
ARIMA seasonal naive baseline for demand forecasting
it shows the minimum acceptable performance bar
Any model deployed to production must surpass the baseline
"""

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class SeasonalNaiveBaseline:
    """
    Seasonal naive baseline: predicts the same value as the same day last week (7 day lag)
    """

    def __init__(self, seasonal_period: int = 7) -> None:
        self.seasonal_period = seasonal_period
        self._last_known: dict[int, dict] = {}  # store_id → {date → sales}
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Stores the last seasonal_period days of sales per store.
        """
        train_df = train_df.sort_values(["store", "date"])

        for store_id, group in train_df.groupby("store"):
            self._last_known[int(store_id)] = dict(
                zip(pd.to_datetime(group["date"]), group["sales"])
            )
        self._is_fitted = True
        logger.info(
            "Baseline fitted on {stores} stores",
            stores=len(self._last_known),
        )

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon: int = 14,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict by looking up sales from seasonal_period days ago.

        For dates where no historical value exists, use the store mean.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        features_df = features_df.sort_values("date").copy()
        features_df["date"] = pd.to_datetime(features_df["date"])

        predictions = []

        for _, row in features_df.iterrows():
            store_id = int(row["store"])
            current_date = pd.Timestamp(row["date"])
            lag_date = current_date - pd.Timedelta(days=self.seasonal_period)

            store_history = self._last_known.get(store_id, {})
            lag_value = store_history.get(lag_date)

            if lag_value is None:
                lag_value = (
                    np.mean(list(store_history.values())) if store_history else 0.0
                )

            predictions.append(max(0.0, float(lag_value)))

        point = np.array(predictions)
        lower = point * 0.8
        upper = point * 1.2

        return point, lower, upper
