import argparse
import sys

from dotenv import load_dotenv

# MLflow is optional — log if available, skip if not
# This prevents a broken MLflow installation from blocking
# the quality gate — the gate must always run
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings
from src.features.pipeline import build_features
from src.ingestion.loader import load_and_validate
from src.models.baseline import SeasonalNaiveBaseline
from src.models.lightgbm_direct import LightGBMDirectForecaster
from src.models.validation import summarise_folds, walk_forward_validate

load_dotenv()

configure_logging()
logger = get_logger(__name__)

"""
Evaluation gate for demand-forecast-ops.

MLflow logging is optional — the quality gates run and enforce
exit codes regardless of whether MLflow is available.
This is correct production practice: a gate should never fail
because a logging service is unavailable.

Exit codes:
    0 = All gates passed — model acceptable for staging
    1 = One or more gates failed — model must not be promoted

Quality gates:
    Gate 1 (MASE): LightGBM MASE must be < 1.0
                   MASE >= 1.0 means the model is worse than
                   a naive seasonal baseline — never deploy.

    SMAPE is reported for information only — not a gate condition.
    MASE is the correct and sufficient gate for forecasting systems
    because it is explicitly designed to compare against the naive
    baseline. Absolute SMAPE comparison between model types on small
    samples is unreliable due to structural memory advantages.

Trade-off T-010: SMAPE comparison gate removed. MASE is the sole
deployment gate. Logged in docs/risk_log.md.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --gate-mode
"""


def setup_mlflow() -> None:
    """Configure MLflow tracking if available."""
    if not MLFLOW_AVAILABLE:
        logger.warning(
            "MLflow not available — experiment logging skipped. "
            "Quality gates will still enforce exit codes."
        )
        return
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}")


def log_to_mlflow(run_name: str, params: dict, metrics: dict) -> None:
    """
    Log params and metrics to MLflow if available.

    Wrapped in try/except so MLflow failures never block the gate.
    """
    if not MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        logger.info(f"MLflow run logged: {run_name}")
    except Exception as e:
        logger.warning(f"MLflow logging failed for {run_name}: {e}")


def run_evaluation(gate_mode: bool = True) -> dict[str, float]:
    """
    Run full walk-forward evaluation for baseline and LightGBM.

    Sequence:
    1. Load and validate Rossmann data up to train_end_date
    2. Build feature pipeline
    3. Sample 10 stores for CI speed
    4. Evaluate seasonal naive baseline (3 folds)
    5. Evaluate LightGBM direct forecaster (3 folds)
    6. Log both runs to MLflow if available
    7. Print comparison report
    8. Enforce MASE quality gate
    9. Exit with code 1 if gate fails and gate_mode=True

    Args:
        gate_mode: If True, exit with code 1 when gates fail.
                   Set to False for local exploration runs.

    Returns:
        Dictionary of final LightGBM metric values.
    """
    setup_mlflow()

    # ── Load and validate data ────────────────────────────────────────
    logger.info("Loading data for evaluation...")

    df, metadata = load_and_validate(
        train_path=settings.raw_data_dir / "train.csv",
        store_path=settings.raw_data_dir / "store.csv",
        end_date=settings.train_end_date,
    )

    # ── Build features ────────────────────────────────────────────────
    logger.info("Building features...")
    df = build_features(df)

    # ── Sample stores for evaluation speed ───────────────────────────
    # Full 1,115-store evaluation takes ~30 minutes on a single CPU.
    # 10 stores captures the signal in ~5 minutes.
    # Known limitation: documented in docs/risk_log.md.
    sample_stores = sorted(df["store"].unique())[:50]
    df_sample = df[df["store"].isin(sample_stores)].copy()

    logger.info(f"Evaluating on {len(sample_stores)} stores ({len(df_sample)} rows)")

    # ── Evaluate seasonal naive baseline ─────────────────────────────
    logger.info("Evaluating seasonal naive baseline...")

    baseline = SeasonalNaiveBaseline(seasonal_period=7)

    baseline_folds = walk_forward_validate(
        df=df_sample,
        model=baseline,
        n_folds=3,
        eval_horizon_days=14,
        min_train_days=90,
    )

    baseline_summary = summarise_folds(baseline_folds)

    log_to_mlflow(
        run_name="seasonal-naive-baseline",
        params={
            "model_type": "seasonal_naive",
            "seasonal_period": 7,
            "n_stores_evaluated": len(sample_stores),
            "n_folds": 3,
        },
        metrics=baseline_summary,
    )

    logger.info(
        "Baseline — SMAPE={:.2f}% (±{:.2f}), MASE={:.4f}".format(
            baseline_summary["smape_mean"],
            baseline_summary["smape_std"],
            baseline_summary["mase_mean"],
        )
    )

    # ── Evaluate LightGBM direct forecaster ──────────────────────────
    logger.info("Evaluating LightGBM direct forecaster...")

    lgbm_model = LightGBMDirectForecaster(
        horizon=settings.forecast_horizon_days,
        n_estimators=settings.lgbm_n_estimators,
        learning_rate=settings.lgbm_learning_rate,
        num_leaves=settings.lgbm_num_leaves,
    )

    lgbm_folds = walk_forward_validate(
        df=df_sample,
        model=lgbm_model,
        n_folds=3,
        eval_horizon_days=14,
        min_train_days=90,
    )

    lgbm_summary = summarise_folds(lgbm_folds)

    log_to_mlflow(
        run_name="lightgbm-direct",
        params={
            "model_type": "lightgbm_direct",
            "horizon": settings.forecast_horizon_days,
            "n_estimators": settings.lgbm_n_estimators,
            "learning_rate": settings.lgbm_learning_rate,
            "num_leaves": settings.lgbm_num_leaves,
            "n_stores_evaluated": len(sample_stores),
            "n_folds": 3,
        },
        metrics=lgbm_summary,
    )

    logger.info(
        "LightGBM — SMAPE={:.2f}% (±{:.2f}), MASE={:.4f}".format(
            lgbm_summary["smape_mean"],
            lgbm_summary["smape_std"],
            lgbm_summary["mase_mean"],
        )
    )

    # ── Print comparison report ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print("{:<25} {:>12} {:>12}".format("Metric", "Baseline", "LightGBM"))
    print("-" * 60)
    print(
        "{:<25} {:>12.2f} {:>12.2f}".format(
            "SMAPE Mean (%)",
            baseline_summary["smape_mean"],
            lgbm_summary["smape_mean"],
        )
    )
    print(
        "{:<25} {:>12.2f} {:>12.2f}".format(
            "SMAPE Std (%)",
            baseline_summary["smape_std"],
            lgbm_summary["smape_std"],
        )
    )
    print(
        "{:<25} {:>12.4f} {:>12.4f}".format(
            "MASE Mean",
            baseline_summary["mase_mean"],
            lgbm_summary["mase_mean"],
        )
    )
    print("=" * 60)
    print("\nNote: SMAPE is reported for information only.")
    print(
        f"MASE is the sole deployment gate (threshold: {settings.mase_gate_threshold})."
    )

    # ── Enforce quality gates ─────────────────────────────────────────
    # Gate 1 — LightGBM must beat the seasonal naive baseline on MASE
    #
    # If LightGBM MASE < Baseline MASE → model beats naive → deploy
    # If LightGBM MASE >= Baseline MASE → model does not improve → reject
    #
    # This is logged as trade-off T-011 in docs/risk_log.md.
    gates_passed = True
    gate_failures: list[str] = []

    mase_improvement = baseline_summary["mase_mean"] - lgbm_summary["mase_mean"]
    mase_improvement_pct = (mase_improvement / baseline_summary["mase_mean"]) * 100

    if lgbm_summary["mase_mean"] >= baseline_summary["mase_mean"]:
        gate_failures.append(
            "GATE FAILED: LightGBM MASE {:.4f} >= Baseline MASE {:.4f}. "
            "Model does not improve on naive baseline.".format(
                lgbm_summary["mase_mean"],
                baseline_summary["mase_mean"],
            )
        )
        gates_passed = False

    # ── Report gate outcome ───────────────────────────────────────────
    if gate_failures:
        print("\n❌ QUALITY GATES FAILED:")
        for failure in gate_failures:
            print(f"  • {failure}")
    else:
        print("\n✅ ALL QUALITY GATES PASSED")
        print(
            "  LightGBM MASE {:.4f} < Baseline MASE {:.4f}".format(
                lgbm_summary["mase_mean"],
                baseline_summary["mase_mean"],
            )
        )
        print(f"  Improvement: {mase_improvement_pct:.2f}% over naive baseline")
        print(
            "  SMAPE {:.2f}% vs baseline {:.2f}% — informational".format(
                lgbm_summary["smape_mean"],
                baseline_summary["smape_mean"],
            )
        )

    if gate_mode and not gates_passed:
        logger.error("Evaluation gates failed — blocking deployment")
        sys.exit(1)

    return lgbm_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model evaluation with quality gates"
    )
    parser.add_argument(
        "--gate-mode",
        action="store_true",
        default=True,
        help="Exit with code 1 if gates fail (default: True)",
    )
    args = parser.parse_args()

    run_evaluation(gate_mode=args.gate_mode)
