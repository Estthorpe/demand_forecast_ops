# Model Card — LightGBM Demand Forecaster

## Model Details

| Field            | Value                                     |
|------------------|-------------------------------------------|
| Model type       | LightGBM direct multi-step forecaster     |
| Framework        | LightGBM 4.x, scikit-learn                |
| Horizon          | 1–14 days (configurable)                  |
| Granularity      | Daily, per store                          |
| Output           | Point forecast + 90% prediction intervals |
| Training data    | Rossmann Store Sales, 2013–2015           |
| Stores covered   |  1,115                                    |
| Training records | ~982,644                                  |

## Intended Use

**Primary use case:** Retail demand forecasting for replenishment planning at Rossmann stores.

**Intended users:** Supply chain analysts, operations teams, automated replenishment systems.

**Out-of-scope uses:**
- Forecasting for stores not in the Rossmann dataset
- Horizons beyond 14 days without revalidation
- Financial planning or investor-facing projections

## Performance

| Metric                         | Value |
|--------------------------------|--------|
| MASE (LightGBM)                | 1.0338 |
| MASE (Seasonal Naive Baseline) | 1.2956 |
| Improvement over baseline      | 20.21% |

Evaluated using walk-forward expanding window validation.
Each fold trains on all data up to a cutoff and tests on the next window.
This prevents data leakage and reflects real deployment conditions.

## Features

| Category | Features                                                         |
|----------|------------------------------------------------------------------|
| Calendar | day_of_week, month, day_of_month, is_weekend, cyclical encodings |
| Lag      | sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28             |
| Rolling  | sales_rolling_mean_7, sales_rolling_std_7, sales_rolling_mean_28 |
| Store    | store_type, assortment, competition_distance, promo2             |
| Event    | promo, school_holiday, state_holiday                             |

**Note (T-012):** Lag and rolling features are set to 0 at inference time due to
groupby.transform deadlock in uvicorn async on Windows. This reduces accuracy
vs training but prevents runtime errors. See risk_log.md.

## Limitations

- **Training-serving skew:** Lag/rolling features are 0 at inference, non-zero at training. This is a known accuracy trade-off documented in T-012.
- **Temporal scope:** Trained on 2013–2015 data. Performance on post-2015 patterns is not validated.
- **Store coverage:** Only valid for store IDs 1–1115 in the Rossmann dataset.
- **Holiday encoding:** State holidays are encoded as categorical strings. Unseen holiday types default to -1.
- **Closed stores:** Store 622 has no sales data and is excluded from evaluation.

## Monitoring

The model is monitored via `src/monitoring/` with:
- PSI drift detection on input features (threshold: 0.2)
- Forecast distribution health checks
- MASE degradation tracking (critical threshold: 2.0)
- Automatic retraining flag when thresholds are breached

## Ethical Considerations

- This model forecasts demand for retail supply chain planning only
- Predictions should not be used for workforce planning or staffing decisions
- Model outputs are probabilistic — 90% intervals should be communicated clearly
- Retraining should be triggered when drift is detected to maintain accuracy

## Training Infrastructure

- Python 3.12
- LightGBM with quantile regression for prediction intervals
- MLflow experiment tracking via DagsHub
- DVC for dataset and artefact versioning
