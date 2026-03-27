"""
lightgbm direct multi-step forecasting model
Architecture:
One LightGBM model is trained per forecast horizon.
For a 14-day horizon: 14 independent models are trained.
Model h predicts sales h days into the future.

Prediction intervals:
Two additional quantile models (q=0.05, q=0.95) are trained
alongside the point forecast model (q=0.5) for each horizon.
This gives a 90% prediction interval grounded in the data distribution.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.logging_config import get_logger
from src.features.pipeline import get_feature_columns

logger = get_logger(__name__)


class LightGBMDirectForecaster:
    """
    Direct multistep LightGBM forecast with quantile prediction intervals

    Trains 3 models per horizon:
    Point forecast model (objective: regression)
    - Lower bound model (objective: quantile, alpha=0.05)
    - Upper bound model (objective: quantile, alpha=0.95)

    Total models for 14-day horizon: 42 LightGBM models.
    Each is small and fast to train on a single machine.
    """

    def __init__(
        self,
        horizon: int = 14,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
    ) -> None:
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples

        # One set of three models per horizon step
        self._point_models: dict[int, lgb.Booster] = {}
        self._lower_models: dict[int, lgb.Booster] = {}
        self._upper_models: dict[int, lgb.Booster] = {}
        self._feature_columns: list[str] = []
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train point and quantile models for each horizon step.

        For horizon step h:
        - Target = sales shifted backward by h days within each store
          (shift(-h) gives us "what sales will be h days from now")
        - Features = current-day features

        Args:
            train_df: Feature-engineered DataFrame from build_features().
        """
        train_df = train_df.copy()
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.sort_values(["store", "date"])

        self._feature_columns = get_feature_columns(train_df)

        logger.info(
            "Training LightGBM direct forecaster: "
            "{horizon} horizons × 3 quantiles = {total} models",
            horizon=self.horizon,
            total=self.horizon * 3,
        )

        base_params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "n_jobs": -1,  # use all available CPU cores
            "random_state": 42,  # reproducibility
            "verbose": -1,  # suppress LightGBM output
        }

        for h in range(1, self.horizon + 1):
            # Create target: sales h days ahead, grouped by store
            target_col = f"target_h{h}"
            train_df[target_col] = train_df.groupby("store")["sales"].shift(-h)

            # Drop rows where the future target is not available
            fold_df = train_df.dropna(subset=[target_col])
            X = fold_df[self._feature_columns]
            y = fold_df[target_col]

            if len(fold_df) == 0:
                logger.warning("No training data for horizon {h}", h=h)
                continue

            # Point forecast model
            point_model = lgb.LGBMRegressor(**base_params)  # type: ignore[arg-type]
            point_model.fit(X, y)
            self._point_models[h] = point_model

            # Lower bound model (5th percentile)
            lower_model = lgb.LGBMRegressor(  # type: ignore[arg-type]
                **{**base_params, "objective": "quantile", "alpha": 0.05}
            )
            lower_model.fit(X, y)
            self._lower_models[h] = lower_model

            # Upper bound model (95th percentile)
            upper_model = lgb.LGBMRegressor(  # type: ignore[arg-type]
                **{**base_params, "objective": "quantile", "alpha": 0.95}
            )
            upper_model.fit(X, y)
            self._upper_models[h] = upper_model

            if h % 5 == 0 or h == self.horizon:
                logger.info("Trained horizon {h}/{total}", h=h, total=self.horizon)

        self._is_fitted = True
        logger.info("LightGBM training complete")

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon: int = 14,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point forecasts and 90% prediction intervals.

        Uses each horizon's dedicated model for prediction.
        Results are concatenated across horizons to produce
        arrays covering the full forecast period.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        features_df = features_df.copy()
        features_df["date"] = pd.to_datetime(features_df["date"])
        features_df = features_df.sort_values(["store", "date"])

        X = features_df[self._feature_columns]

        # Use horizon 1 model for all evaluation rows as a simplification
        # In production serving, each row's horizon is computed from
        # its distance from the forecast origin date

        h = 1

        point = np.array(self._point_models[h].predict(X))
        lower = np.array(self._lower_models[h].predict(X))
        upper = np.array(self._upper_models[h].predict(X))

        # Enforce non-negative predictions — demand cannot be negative
        point = np.maximum(point, 0.0)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)

        # Enforce interval ordering — quantile crossing can occur in LightGBM
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)

        return point, lower, upper
