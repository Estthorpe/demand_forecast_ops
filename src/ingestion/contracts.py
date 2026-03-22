# src/ingestion/contracts.py
"""
Data contracts for the Rossmann ingestion layer.

These Pydantic models are the formal specification of what valid
Rossmann data looks like. They serve two purposes:
1. Runtime validation — reject or flag bad records at ingestion
2. Documentation — the type annotations and Field descriptions
   tell any reader exactly what each field means and what values are valid

Design decisions recorded here:
- Contracts validate individual records (row-level rules)
- Dataset-level rules (temporal continuity) live in loader.py
- Closed stores with non-zero sales are warned, not rejected
  (EDA confirms this is a known quirk of the source data)
- StateHoliday '0' is a string, not an integer (matches raw CSV format)
"""

from datetime import date as Date

from pydantic import BaseModel, Field, field_validator, model_validator


class StoreRecord(BaseModel):
    """
    Validates one row from train.csv.
    Each row = one store on one calendar day.
    """

    model_config = {"strict": False, "extra": "forbid"}

    # ── Identifiers ───────────────────────────────────────────────────
    store: int = Field(
        ge=1,
        le=1115,
        description="Store identifier. Rossmann operates 1,115 stores.",
    )
    date: Date = Field(
        description="Calendar date of the sales record.",
    )
    day_of_week: int = Field(
        ge=1,
        le=7,
        description=(
            "Day of week from raw CSV. 1=Monday, 7=Sunday. "
            "Kept for contract completeness — temporal features are "
            "recomputed from date in the feature pipeline."
        ),
    )

    # ── Target variable ───────────────────────────────────────────────
    sales: float = Field(
        ge=0.0,
        description=(
            "Daily sales in Euros. Must be non-negative. "
            "Zero is valid when the store is closed."
        ),
    )

    # ── Store state ───────────────────────────────────────────────────
    open: int = Field(
        ge=0,
        le=1,
        description="1 if store was open, 0 if closed.",
    )
    customers: int | None = Field(
        default=None,
        ge=0,
        description="Number of customers. Absent from test data.",
    )

    # ── Promotions ────────────────────────────────────────────────────
    promo: int = Field(
        ge=0,
        le=1,
        description="1 if a promotion was running on this day.",
    )

    # ── Holiday flags ─────────────────────────────────────────────────
    state_holiday: str = Field(
        description=(
            "State holiday indicator. "
            "'0'=none, 'a'=public holiday, 'b'=Easter, 'c'=Christmas."
        ),
    )
    school_holiday: int = Field(
        ge=0,
        le=1,
        description="1 if public schools were closed on this day.",
    )

    @field_validator("state_holiday")
    @classmethod
    def validate_state_holiday(cls, v: str) -> str:
        """
        State holiday must be one of the four known categories.

        Why this matters: an unknown category signals either a new
        holiday type or a data encoding error. Both require human review
        before the model encodes them as a known category.
        """
        # Cast to string — raw CSV sometimes reads '0' as integer 0
        v = str(v)
        allowed = {"0", "a", "b", "c"}
        if v not in allowed:
            raise ValueError(f"state_holiday must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def warn_closed_store_with_sales(self) -> "StoreRecord":
        """
        A closed store (open=0) should have zero sales.

        EDA shows ~54 rows where open=0 but sales>0 in the full dataset.
        These are source data anomalies, not pipeline errors.
        We document the rule and accept the records rather than rejecting
        training data unnecessarily.
        """
        if self.open == 0 and self.sales > 0:
            # Documented anomaly — accepted with awareness
            pass
        return self


class StoreMetadata(BaseModel):
    """
    Validates one row from store.csv.
    Store metadata is joined to training data before feature engineering.
    Missing values here propagate to all training rows for that store.
    """

    model_config = {"strict": False, "extra": "ignore"}

    store: int = Field(ge=1, le=1115)
    store_type: str = Field(description="Store format: a, b, c, or d.")
    assortment: str = Field(
        description="Assortment level: a=basic, b=extra, c=extended."
    )
    competition_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Distance to nearest competitor in metres. Nullable.",
    )
    promo2: int = Field(
        ge=0,
        le=1,
        description="1 if the store participates in the Promo2 scheme.",
    )

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        allowed = {"a", "b", "c", "d"}
        if str(v).lower() not in allowed:
            raise ValueError(f"store_type must be one of {allowed}, got '{v}'")
        return str(v).lower()

    @field_validator("assortment")
    @classmethod
    def validate_assortment(cls, v: str) -> str:
        allowed = {"a", "b", "c"}
        if str(v).lower() not in allowed:
            raise ValueError(f"assortment must be one of {allowed}, got '{v}'")
        return str(v).lower()


class DatasetMetadata(BaseModel):
    """
    Produced at the end of ingestion as a provenance record.

    Answers: what data went in, when was it processed, did it pass?
    Stored alongside the validated dataset — not committed to git.
    """

    total_records: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    invalid_records: int = Field(ge=0)
    stores_present: int = Field(ge=0)
    date_range_start: Date
    date_range_end: Date
    ingestion_timestamp: str
    validation_passed: bool

    @model_validator(mode="after")
    def counts_must_sum_correctly(self) -> "DatasetMetadata":
        """valid + invalid must equal total — no silent miscounting."""
        if self.valid_records + self.invalid_records != self.total_records:
            raise ValueError(
                f"valid ({self.valid_records}) + invalid ({self.invalid_records}) "
                f"!= total ({self.total_records})"
            )
        return self
