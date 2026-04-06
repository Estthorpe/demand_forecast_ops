"""
Full training pipeline for demand-forecast-ops
Trains LightGBM direct froecaster on the full training dataet
and saves  the model artefact to models/ for serving

Outputs:
models/lgbm_forecaster.joblob  -trained model
-models/feature_columns.joblib -feature columns list
modles/training_metadata.json - provenance record
"""

import json
from datetime import UTC, datetime

import joblib
from dotenv import load_dotenv

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.features.pipeline import build_features, get_feature_columns
from src.ingestion.loader import load_and_validate
from src.models.lightgbm_direct import LightGBMDirectForecaster

load_dotenv()
configure_logging()
logger = get_logger(__name__)


def train_and_save() -> None:
    """Train the LightGBM direct forecaster and save the model artefact and metadata"""

    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load and validate the training data ----------
    logger.info("Loading training data....")
    df, metadata = load_and_validate(
        train_path=settings.raw_data_dir / "train.csv",
        store_path=settings.raw_data_dir / "store.csv",
        end_date=settings.train_end_date,
    )

    # ---------- Build features ----------
    logger.info("Building features...")
    df = build_features(df)

    feature_columns = get_feature_columns(df)
    logger.info(f"Training on {len(feature_columns)} features")

    # ---------- Train the model ----------
    logger.info("Training LightGBM direct forecaster...")
    model = LightGBMDirectForecaster(
        horizon=settings.forecast_horizon_days,
        n_estimators=settings.lgbm_n_estimators,
        learning_rate=settings.lgbm_learning_rate,
        num_leaves=settings.lgbm_num_leaves,
    )
    model.fit(df)

    # ---------- Save the model artefact ----------
    model_path = model_dir / "lgbm_forecaster.joblib"
    columns_path = model_dir / "feature_columns.joblib"
    metadata_path = model_dir / "training_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, columns_path)

    training_metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "train_end_date": settings.train_end_date,
        "n_features": len(feature_columns),
        "n_training_rows": len(df),
        "n_estimators": settings.lgbm_n_estimators,
        "horizon_days": settings.forecast_horizon_days,
        "model_path": str(model_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(training_metadata, f, indent=2)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training metadata saved to {metadata_path}")


if __name__ == "__main__":
    train_and_save()
