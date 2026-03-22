# used to confirm if  data contracts are working before training

import json

from dotenv import load_dotenv

from src.config.logging_config import configure_logging
from src.config.settings import settings
from src.ingestion.loader import load_and_validate

load_dotenv()

configure_logging()


def main() -> None:
    df, metadata = load_and_validate(
        train_path=settings.raw_data_dir / "train.csv",
        store_path=settings.raw_data_dir / "store.csv",
        end_date=settings.train_end_date,
    )

    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(json.dumps(metadata.model_dump(), indent=2, default=str))
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
