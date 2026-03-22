# scripts/explore_data.py
"""
Exploratory analysis of the Rossmann dataset.
It is a one-time analysis tool — do not import it from application code.
"""

import pandas as pd
from dotenv import load_dotenv

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings

load_dotenv()
configure_logging()
logger = get_logger(__name__)


def explore() -> None:
    """Print key statistics that inform data contract design."""

    train_path = settings.raw_data_dir / "train.csv"
    store_path = settings.raw_data_dir / "store.csv"

    logger.info("Loading data for exploration...")

    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)

    print("\n" + "=" * 60)
    print("TRAIN DATA — SHAPE AND DATE RANGE")
    print("=" * 60)
    print(f"Rows:        {len(train):,}")
    print(f"Columns:     {list(train.columns)}")
    print(f"Date range:  {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"Stores:      {train['Store'].nunique()}")

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(train.isnull().sum())

    print("\n" + "=" * 60)
    print("SALES STATISTICS")
    print("=" * 60)
    print(train["Sales"].describe())

    print("\n" + "=" * 60)
    print("CONTRACT-INFORMING OBSERVATIONS")
    print("=" * 60)

    # 1. Negative sales?
    neg = (train["Sales"] < 0).sum()
    print(f"Negative sales rows:                {neg}")

    # 2. Closed stores with non-zero sales?
    closed_with_sales = ((train["Open"] == 0) & (train["Sales"] > 0)).sum()
    print(f"Closed store but Sales > 0:         {closed_with_sales}")

    # 3. Open stores with zero sales?
    open_zero = ((train["Open"] == 1) & (train["Sales"] == 0)).sum()
    print(f"Open store but Sales = 0:           {open_zero}")

    # 4. Store ID range
    print(
        f"Store ID range:                     {train['Store'].min()} → {train['Store'].max()}"
    )

    # 5. State holiday categories
    print(
        f"StateHoliday categories:            {sorted(train['StateHoliday'].unique())}"
    )

    # 6. Date gaps — check Store 1 as a sample
    store1 = train[train["Store"] == 1].sort_values("Date")
    gaps = store1["Date"].diff().dt.days.dropna()
    print(f"Store 1 max date gap (days):        {int(gaps.max())}")
    print(f"Store 1 gaps > 7 days:              {(gaps > 7).sum()}")

    print("\n" + "=" * 60)
    print("STORE METADATA")
    print("=" * 60)
    print(f"Rows:        {len(store)}")
    print(f"Missing:\n{store.isnull().sum()}")
    print(f"StoreType:   {store['StoreType'].value_counts().to_dict()}")
    print(f"Assortment:  {store['Assortment'].value_counts().to_dict()}")


if __name__ == "__main__":
    explore()
