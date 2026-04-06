# src/serving/predictor.py
"""
Model loading and inference logic for the forecast API.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.features.calendar import add_calendar_features
from src.models.lightgbm_direct import LightGBMDirectForecaster
from src.serving.schemas import DayForecast

logger = get_logger(__name__)

# Encoding maps for categorical columns
# Must match the encoding applied in src/features/pipeline.py
STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}

# Lag and rolling feature columns the model was trained on
# These must exist in the inference DataFrame even with value 0
# If you add new lag features to lag_features.py, add them here too
LAG_AND_ROLLING_COLUMNS = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_std_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_28",
]


class ForecastPredictor:
    """
    Wraps the trained LightGBM model for inference.

    Lifecycle:
    1. Instantiated once at module import time (singleton)
    2. load_model() called once at FastAPI startup
    3. predict() called for each /forecast request
    4. Never retrained — model updates require a new deployment
    """

    def __init__(self) -> None:
        self._model: LightGBMDirectForecaster | None = None
        self._feature_columns: list[str] | None = None
        self._model_version: str = "not_loaded"
        self._is_loaded: bool = False

    def load_model(
        self,
        model_path: Path | None = None,
        columns_path: Path | None = None,
    ) -> None:
        """
        Load model artefacts from disk.

        Called once at FastAPI startup via the lifespan context manager.
        Raises FileNotFoundError if artefacts are missing — this causes
        /ready to return 503, which is the correct operational response.
        """
        model_path = model_path or (settings.model_dir / "lgbm_forecaster.joblib")
        columns_path = columns_path or (settings.model_dir / "feature_columns.joblib")
        metadata_path = settings.model_dir / "training_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at {model_path}. "
                "Run python scripts/train.py first."
            )

        logger.info(f"Loading model from {model_path}")
        self._model = joblib.load(model_path)
        self._feature_columns = joblib.load(columns_path)

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            self._model_version = meta.get("trained_at", "unknown")[:10]
        else:
            self._model_version = "unknown"

        self._is_loaded = True
        logger.info(
            f"Model loaded — version {self._model_version}, {len(self._feature_columns) if self._feature_columns else 0} features"
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    def _build_inference_features(
        self,
        rows: list[dict],
    ) -> pd.DataFrame:
        """
        Build features for inference.

        Steps:
        1. Build calendar features (safe — no groupby)
        2. Encode categorical columns to match training
        3. Add all lag/rolling columns as 0 (cold-start)
        4. Fill any remaining NaN with 0

        The resulting DataFrame contains every column the model
        was trained on, so model.predict() never raises KeyError.
        """
        inference_df = pd.DataFrame(rows)
        inference_df["date"] = pd.to_datetime(inference_df["date"])

        # Step 1: Calendar features — safe, no groupby transform
        inference_df = add_calendar_features(inference_df)

        # Step 2: Encode categorical columns
        if "store_type" in inference_df.columns:
            inference_df["store_type"] = (
                inference_df["store_type"].map(STORE_TYPE_MAP).fillna(-1).astype(int)
            )
        if "assortment" in inference_df.columns:
            inference_df["assortment"] = (
                inference_df["assortment"].map(ASSORTMENT_MAP).fillna(-1).astype(int)
            )

        # Step 3: Add lag and rolling columns as 0
        # The model was trained with these columns — they must be
        # present in the inference DataFrame even though their
        # values are 0 (no sales history available at inference time)
        for col in LAG_AND_ROLLING_COLUMNS:
            if col not in inference_df.columns:
                inference_df[col] = 0.0

        # Step 4: Fill any remaining NaN with 0
        inference_df = inference_df.fillna(0)

        return inference_df

    def predict(
        self,
        store_id: int,
        start_date: date,
        horizon_days: int,
        promo: int = 0,
        school_holiday: int = 0,
        state_holiday: str = "0",
    ) -> list[DayForecast]:
        """
        Generate forecasts for a store over the requested horizon.

        Process:
        1. Build a synthetic row for each forecast date
        2. Apply _build_inference_features() — calendar + zero lags
        3. Select exactly the columns the model was trained on
        4. Call model.predict() to get point and interval forecasts
        5. Return typed DayForecast objects

        Args:
            store_id:       Store to forecast for (1-1115).
            start_date:     First date in the forecast horizon.
            horizon_days:   Number of days to forecast ahead.
            promo:          Promotion flag (0 or 1).
            school_holiday: School holiday flag (0 or 1).
            state_holiday:  State holiday category string.

        Returns:
            List of DayForecast objects, one per day in the horizon.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before predict()."
            )

        # ── Build raw rows ────────────────────────────────────────────
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]

        rows = []
        for forecast_date in forecast_dates:
            rows.append(
                {
                    "store": store_id,
                    "date": forecast_date,
                    "sales": 0.0,
                    "open": 1,
                    "promo": promo,
                    "school_holiday": school_holiday,
                    "state_holiday": state_holiday,
                    "customers": None,
                    "store_type": "a",
                    "assortment": "a",
                    "competition_distance": 1000.0,
                    "promo2": 0,
                }
            )

        # ── Build inference features ──────────────────────────────────
        inference_df = self._build_inference_features(rows)

        logger.info(
            f"Inference features built: {len(inference_df)} rows, {len(inference_df.columns)} columns"
        )

        # ── Select exactly the training feature columns ───────────────
        # This is the critical alignment step.
        # self._feature_columns is the exact list saved during training.
        # We must pass the model exactly these columns in exactly this order.
        # Any column in training but missing from inference → add as 0.
        # Any column in inference but not in training → ignore.
        if self._feature_columns is not None:
            # Add any training columns still missing from inference df
            for col in self._feature_columns:
                if col not in inference_df.columns:
                    logger.warning(
                        f"Training column '{col}' missing from inference — "
                        "adding as 0"
                    )
                    inference_df[col] = 0.0

            # Select only training columns in the correct order
            X = inference_df[self._feature_columns].fillna(0)
        else:
            X = inference_df.fillna(0)

        logger.info(f"Passing {len(X.columns)} features to model")

        # ── Generate predictions ──────────────────────────────────────
        point, lower, upper = self._model.predict(X, horizon=horizon_days)

        # ── Build response objects ────────────────────────────────────
        results: list[DayForecast] = []
        for i, forecast_date in enumerate(forecast_dates):
            if i < len(point):
                results.append(
                    DayForecast(
                        forecast_date=forecast_date,
                        point_forecast=round(float(point[i]), 2),
                        lower_90=round(float(lower[i]), 2),
                        upper_90=round(float(upper[i]), 2),
                    )
                )

        logger.info(f"Generated {len(results)} forecasts for store {store_id}")

        return results


# Single predictor instance shared across all requests
predictor = ForecastPredictor()
