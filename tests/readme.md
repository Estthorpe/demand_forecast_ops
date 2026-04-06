# demand-forecast-ops

[![CI Pipeline](https://github.com/Estthorpe/demand_forecast_ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Estthorpe/demand_forecast_ops/actions/workflows/ci.yml)

Production demand forecasting system for retail supply chain.
Portfolio Project P5 of 8 — AI Engineering Portfolio 2025–2026.

## Business Problem
Retailers lose revenue through stockouts and waste. A demand forecast
system with calibrated uncertainty estimates enables replenishment
decisions that balance both risks.

## ML Approach
- Baseline: Seasonal Naive (same day last week)
- Model: LightGBM Direct Multi-Step (14 horizon models)
- Validation: Walk-forward expanding window (3 folds)
- Metrics: SMAPE (informational), MASE (deployment gate)

## Performance
| Metric | Baseline | LightGBM | Improvement |
|--------|----------|----------|-------------|
| SMAPE  | 78.25%   | 75.34%   | —           |
| MASE   | 1.2956   | 1.0338   | 20.21%      |

## Quick Start
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/evaluate.py

## Project Structure
src/           Application source code
tests/         Unit, integration, scenario tests
configs/       DVC parameters
scripts/       Training, evaluation, data scripts
docs/          Architecture, runbook, risk log
