"""
Data loading and validation for the Rossmann dataset.

Responsibilities:
- Load raw CSV files from data/raw/
- Apply Pydantic contracts to every record
- Check temporal continuity per store
- Merge store metadata onto training records
- Return a clean DataFrame ready for feature engineering
- Produce a DatasetMetadata provenance record
"""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.config.logging_config import configure_logging, get_logger
from src.ingestion.contracts import DatasetMetadata, StoreMetadata, StoreRecord

configure_logging()
logger = get_logger(__name__)


# Maximum gap in days before a continuity warning is flagged
# Rossmann closes Sundays — a 2-day gap (Sat→Mon) is expected.
# Gaps over 7 days indicate genuinely missing records.

MAX_FAILURE_RATE = 0.01
# Maximum acceptable validation failure rate before we abort.
# 1% means: if more than 1 in 100 records is invalid, stop.
# Training on heavily corrupted data produces silently bad models.
MAX_ALLOWED_GAP_DAYS = 7


def load_and_validate(
    train_path: Path,
    store_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """
    Loads Rossmann data, validate contracts and returns a clean DataFRame
    Returns: Tuple of ( validate DataFRame, DatasetMetadata provenance record)
    """
    logger.info("Loading raw training data from {path}", path=train_path)

    # ______Load CSVs_______________________________________
    train_df = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        low_memory=False,
    )
    store_df = pd.read_csv(store_path)

    logger.info(
        "Loaded {rows:,} rows across {stores} stores",
        rows=len(train_df),
        stores=train_df["Store"].nunique(),
    )

    # __________Apply date filters_____________________________
    if start_date:
        train_df = train_df[train_df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        train_df = train_df[train_df["Date"] <= pd.Timestamp(end_date)]

    logger.info(
        "After date filter: {rows:,} rows ({start} → {end})",
        rows=len(train_df),
        start=train_df["Date"].min().date(),
        end=train_df["Date"].max().date(),
    )

    # ______Validate individual records_______________________
    valid_rows, invalid_count = _validate_records(train_df)
    total = len(train_df)

    # ______Enforce failure rate threshold_____________________
    failure_rate = invalid_count / total if total > 0 else 0.0
    if failure_rate > MAX_FAILURE_RATE:
        raise ValueError(
            f"Validation failure rate {failure_rate:.1%} exceeds "
            f"{MAX_FAILURE_RATE:.0%} threshold. "
            f"{invalid_count:,} of {total:,} records failed. "
            "Inspect the raw data before proceeding."
        )

    if invalid_count > 0:
        logger.warning(
            "{count} records failed validation ({rate:.3%}) — "
            "within threshold, proceeding",
            count=invalid_count,
            rate=failure_rate,
        )

    # ________Check temporal continuity___________________________
    continuity_warnings = _check_temporal_continuity(valid_rows)
    for w in continuity_warnings:
        logger.warning(w)

    if continuity_warnings:
        logger.warning(
            "{n} temporal continuity warnings - review before warnings",
            n=len(continuity_warnings),
        )

    # ________Validate and merge store metadata____________________________
    store_clean = _validate_store_metadata(store_df)

    validated_df = pd.DataFrame([r.model_dump() for r in valid_rows])
    validated_df = validated_df.merge(store_clean, on="store", how="left")

    # ── Build provenance record ───────────────────────────────────────
    metadata = DatasetMetadata(
        total_records=total,
        valid_records=len(valid_rows),
        invalid_records=invalid_count,
        stores_present=int(validated_df["store"].nunique()),
        date_range_start=validated_df["date"].min(),
        date_range_end=validated_df["date"].max(),
        ingestion_timestamp=datetime.now(UTC).isoformat(),
        validation_passed=True,
    )

    logger.info(
        "Ingestion complete — {valid:,} valid records, "
        "{stores} stores, {start} → {end}",
        valid=metadata.valid_records,
        stores=metadata.stores_present,
        start=metadata.date_range_start,
        end=metadata.date_range_end,
    )

    return validated_df, metadata


def _validate_records(
    df: pd.DataFrame,
) -> tuple[list[StoreRecord], int]:
    """
    Apply StoreRecord contract to every row.

    Returns:
        valid_rows:    List of validated StoreRecord objects
        invalid_count: Number of rows that failed validation
    """
    # Rename columns from CSV format to Pydantic field names
    df = df.rename(
        columns={
            "Store": "store",
            "DayOfWeek": "day_of_week",
            "Date": "date",
            "Sales": "sales",
            "Open": "open",
            "Customers": "customers",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
        }
    )

    valid_rows: list[StoreRecord] = []
    invalid_count = 0

    for idx, row in df.iterrows():
        try:
            record = StoreRecord(**row.to_dict())
            valid_rows.append(record)
        except ValidationError as e:
            invalid_count += 1
            # Log first 5 failures in detail — avoid log flooding
            if invalid_count <= 5:
                logger.warning(
                    "Row {idx} failed validation: {errors}",
                    idx=idx,
                    errors=e.errors(),
                )

    return valid_rows, invalid_count


def _validate_store_metadata(store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate store.csv and return a clean DataFrame.

    Missing competition_distance is filled with the median.
    Why median not mean: competition distance is right-skewed.
    The median is more robust to outliers (very remote stores).
    This decision is documented in docs/validation_strategy.md.
    """
    store_df = store_df.rename(
        columns={
            "Store": "store",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
            "Promo2": "promo2",
        }
    )

    median_dist = store_df["competition_distance"].median()
    missing_count = store_df["competition_distance"].isna().sum()

    if missing_count > 0:
        store_df["competition_distance"] = store_df["competition_distance"].fillna(
            median_dist
        )
        logger.info(
            "Filled {n} missing competition_distance values " "with median {m:.0f}m",
            n=missing_count,
            m=median_dist,
        )

    valid_stores = []
    for _, row in store_df.iterrows():
        try:
            record = StoreMetadata(
                store=row["store"],
                store_type=row["store_type"],
                assortment=row["assortment"],
                competition_distance=row["competition_distance"],
                promo2=row["promo2"],
            )
            valid_stores.append(record.model_dump())
        except ValidationError as e:
            logger.warning(
                "Store metadata invalid for store {s}: {e}",
                s=row.get("store"),
                e=e.errors(),
            )

    return pd.DataFrame(valid_stores)


def _check_temporal_continuity(
    records: list[StoreRecord],
) -> list[str]:
    """
    Check for date gaps larger than MAX_ALLOWED_GAP_DAYS per store.
    Returns:
        List of human-readable warning strings.
        Empty list means no gaps found.
    """
    store_dates: dict[int, list] = {}
    for record in records:
        store_dates.setdefault(record.store, []).append(record.date)

    warnings: list[str] = []

    for store_id, dates in store_dates.items():
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i - 1]).days
            if gap > MAX_ALLOWED_GAP_DAYS:
                warnings.append(
                    f"Store {store_id}: {gap}-day gap between "
                    f"{sorted_dates[i - 1]} and {sorted_dates[i]}"
                )

    return warnings
