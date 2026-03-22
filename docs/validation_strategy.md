# Validation Strategy — demand-forecast-ops

## Data Source
Rossmann Store Sales dataset (Kaggle competition).
1,115 stores, daily sales records, January 2013 – July 2015.

## Contract Rules and Their Justification

| Field | Rule | Justification from EDA |
|-------|------|------------------------|
| store | 1 ≤ store ≤ 1115 | Rossmann operates exactly 1,115 stores |
| sales | sales ≥ 0.0 | No negative sales observed in EDA |
| open | 0 or 1 | Binary flag — no other values in data |
| state_holiday | one of {0, a, b, c} | Exhaustive category set from EDA |
| promo | 0 or 1 | Binary flag |

## Known Anomalies (Accepted With Documentation)
- ~54 rows where open=0 but sales>0. Source data quirk. Accepted.
- Missing competition_distance in store.csv. Filled with median.

## Temporal Continuity
- Gap threshold: 7 days
- Rossmann closes Sundays → 2-day gaps (Sat→Mon) are expected
- Gaps > 7 days flag missing records that corrupt lag features

## Train/Validation Split
- Train: all records up to 2015-06-30
- Validation: 2015-07-01 to 2015-07-31
- No shuffling — temporal order is preserved
