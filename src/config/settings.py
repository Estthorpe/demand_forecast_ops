"""
Central settings for demand-forecast-ops
- The type annotations are the documentation
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central Configuration Object
    Load priority (highest to lowest):
    1. Environment variables set in the shell or CI
    2. Values in the .env file
    3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # __________Project Identity______________
    project_name: str = "demand-forecasr-ops"
    project_version: str = "0.1.0"

    # __________Paths_________________________
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # ______MLflow / Dagshub__________________
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "demand-forecast-ops"
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    # ______Data parameters___________________
    train_end_date: str = "2015-06-30"
    validation_end_date: str = "2015-07-31"
    forecast_horizon_days: int = 14
    min_store_history_days: int = 180

    # ________Model Parameters________________
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 20

    # ________Kaggle_________________________
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ________DagsHub________________________
    dagshub_username: str = ""
    dagshub_token: str = ""

    # ________Evaluation gates________________
    # CI pipeline exits with code 1 if these are not met
    # mase_gate_threshold: 1.0 means model must beat the naive baseline
    # A MASE of 1.2 means the model is 20% worse than doing nothing
    mase_gate_threshold: float = 1.0
    smape_gate_threshold: float = 25.0

    # ________Serving_________________________
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_registry_name: str = "demand-forecast"


settings = Settings()
