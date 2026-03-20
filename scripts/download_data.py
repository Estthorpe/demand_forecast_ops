"""Downloads the Rossman Store Sales dataset from Kaggle
Data is tracked by DVC, not git
-Daily sales for 1,115 stores across 2.5 years (2013 - 2015)
"""

import os
import subprocess
import zipfile

from dotenv import load_dotenv

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import settings

load_dotenv()

configure_logging()
logger = get_logger(__name__)


def download_rossmann() -> None:
    """
    Download and extract Rossman dataset to data/raw
    """
    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded - idempotent by design
    if (raw_dir / "train_csv").exists():
        logger.info(
            "Rossman data already exists at {path} - skipping download",
            path=raw_dir,
        )
        return
    # Verify Kaggle credentials
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        raise OSError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file. "
            "Get your token from: kaggle.com → Settings → API → API Tokens"
        )

    logger.info(
        "Downloading Rossmann dataset (authenticated as {user})...",
        user=username,
    )

    # The kaggle CLI picks up KAGGLE_USERNAME and KAGGLE_KEY
    # from the environment automatically
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "rossmann-store-sales",
            "-p",
            str(raw_dir),
        ],
        check=True,  # raises CalledProcessError if kaggle returns non-zero exit code
        env={**os.environ},  # pass full environment including KAGGLE_* vars
    )

    # Extract the downloaded zip
    zip_path = raw_dir / "rossmann-store-sales.zip"
    if zip_path.exists():
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        zip_path.unlink()  # delete zip after extraction — we only need the CSVs
        logger.info("Extraction complete")
    else:
        # Kaggle sometimes names the zip differently — check what arrived
        zips = list(raw_dir.glob("*.zip"))
        if zips:
            logger.info("Extracting {name}...", name=zips[0].name)
            with zipfile.ZipFile(zips[0], "r") as z:
                z.extractall(raw_dir)
            zips[0].unlink()
        else:
            raise FileNotFoundError(
                f"No zip file found in {raw_dir} after download. "
                "Check the kaggle CLI output above for errors."
            )

    logger.info(
        "Rossmann data ready at {path}",
        path=raw_dir,
    )

    # List what we got
    files = list(raw_dir.glob("*.csv"))
    logger.info(
        "Files present: {files}",
        files=[f.name for f in files],
    )


if __name__ == "__main__":
    download_rossmann()
