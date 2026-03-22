# tests/unit/test_contracts.py
"""
Unit tests for Rossmann data contracts.

Each test targets one specific rule in the contract.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.ingestion.contracts import DatasetMetadata, StoreRecord

# ── Helper ─────────────────────────────────────────────────────────────────


def valid_record(**overrides: object) -> dict:
    """
    Returns a dictionary representing a valid StoreRecord.
    Pass keyword arguments to override specific fields for testing.

    Example:
        valid_record(sales=-50.0)   # tests negative sales rejection
        valid_record(store=9999)    # tests out-of-range store ID
    """
    base = {
        "store": 1,
        "date": date(2015, 1, 1),
        "day_of_week": 4,
        "sales": 5263.0,
        "open": 1,
        "customers": 555,
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0,
    }
    base.update(overrides)
    return base


# ── Acceptance tests ───────────────────────────────────────────────────────


def test_valid_record_accepted() -> None:
    """A complete, correctly typed record must pass without error."""
    record = StoreRecord(**valid_record())
    assert record.store == 1
    assert record.sales == 5263.0
    assert record.state_holiday == "0"


def test_boundary_store_ids_accepted() -> None:
    """Store IDs 1 and 1115 are valid boundaries — both must be accepted."""
    StoreRecord(**valid_record(store=1))
    StoreRecord(**valid_record(store=1115))


def test_all_state_holiday_values_accepted() -> None:
    """Each known holiday category must be accepted individually."""
    for value in ["0", "a", "b", "c"]:
        record = StoreRecord(**valid_record(state_holiday=value))
        assert record.state_holiday == value


def test_zero_sales_when_closed_accepted() -> None:
    """A closed store with zero sales is valid — Sunday closures."""
    record = StoreRecord(**valid_record(open=0, sales=0.0))
    assert record.sales == 0.0


# ── Rejection tests ────────────────────────────────────────────────────────


def test_negative_sales_rejected() -> None:
    """
    Negative sales must be rejected.
    Refunds are handled separately — a negative daily total
    indicates a data error, not a legitimate business event.
    """
    with pytest.raises(ValidationError) as exc_info:
        StoreRecord(**valid_record(sales=-1.0))
    assert any("sales" in str(e) for e in exc_info.value.errors())


def test_store_id_above_max_rejected() -> None:
    """Store 1116 does not exist in the Rossmann estate."""
    with pytest.raises(ValidationError):
        StoreRecord(**valid_record(store=1116))


def test_store_id_zero_rejected() -> None:
    """Store ID 0 is not a valid Rossmann store identifier."""
    with pytest.raises(ValidationError):
        StoreRecord(**valid_record(store=0))


def test_invalid_open_flag_rejected() -> None:
    """Open flag is binary — values other than 0 or 1 are invalid."""
    with pytest.raises(ValidationError):
        StoreRecord(**valid_record(open=2))


def test_unknown_state_holiday_rejected() -> None:
    """
    An unknown holiday category must be rejected explicitly.
    It could indicate a new holiday type or an encoding error —
    either requires human review
    """
    with pytest.raises(ValidationError) as exc_info:
        StoreRecord(**valid_record(state_holiday="x"))
    assert "state_holiday" in str(exc_info.value)


def test_invalid_promo_flag_rejected() -> None:
    """Promo is binary — value of 5 is invalid."""
    with pytest.raises(ValidationError):
        StoreRecord(**valid_record(promo=5))


# ── DatasetMetadata tests ──────────────────────────────────────────────────


def test_metadata_counts_must_sum_correctly() -> None:
    """
    valid + invalid must equal total.
    Mismatched counts indicate a bug in the ingestion loop.
    """
    with pytest.raises(ValidationError):
        DatasetMetadata(
            total_records=100,
            valid_records=90,
            invalid_records=5,  # 90 + 5 = 95 ≠ 100 → must fail
            stores_present=50,
            date_range_start=date(2013, 1, 1),
            date_range_end=date(2015, 7, 31),
            ingestion_timestamp="2025-01-01T00:00:00Z",
            validation_passed=True,
        )


def test_metadata_valid_when_counts_correct() -> None:
    """Metadata with correct count arithmetic must be accepted."""
    metadata = DatasetMetadata(
        total_records=100,
        valid_records=95,
        invalid_records=5,
        stores_present=50,
        date_range_start=date(2013, 1, 1),
        date_range_end=date(2015, 7, 31),
        ingestion_timestamp="2025-01-01T00:00:00Z",
        validation_passed=True,
    )
    assert metadata.validation_passed is True
    assert metadata.valid_records == 95
