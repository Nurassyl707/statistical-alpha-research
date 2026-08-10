from pathlib import Path
import time

import pandas as pd

from src.data.prices import download_price_data


# ============================================================
# Configuration
# ============================================================

DEFAULT_UNIVERSE_PATH = "data/raw/universe/sp500.csv"
DEFAULT_OUTPUT_DIR = "data/raw/prices"


# ============================================================
# Price-file validation
# ============================================================

def is_valid_price_file(
    file_path: Path,
    min_rows: int = 500,
) -> bool:
    """
    Check whether a downloaded price CSV is usable.

    A valid price file must:
    - exist
    - contain all required columns
    - contain at least `min_rows` observations
    - contain valid dates
    - contain valid price data
    """

    if not file_path.exists():
        return False

    try:
        data = pd.read_csv(file_path)
    except Exception:
        return False

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    # Check required columns
    if not all(column in data.columns for column in required_columns):
        return False

    # Check minimum number of observations
    if len(data) < min_rows:
        return False

    # Check dates
    try:
        dates = pd.to_datetime(data["Date"])
    except Exception:
        return False

    if dates.isna().any():
        return False

    # Check price columns
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]

    # Reject rows where all price values are missing
    if data[price_columns].isna().all(axis=1).any():
        return False

    return True


# ============================================================
# Universe loading
# ============================================================

def load_universe(
    universe_path: str = DEFAULT_UNIVERSE_PATH,
) -> pd.DataFrame:
    """
    Load the stock universe from CSV.

    Expected columns:
        ticker
        company
        sector
        sub_industry
    """

    file_path = Path(universe_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if "ticker" not in data.columns:
        raise ValueError(
            "Universe file must contain a 'ticker' column."
        )

    # Clean ticker symbols
    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Remove empty tickers
    data = data[data["ticker"] != ""]

    # Remove duplicate tickers
    data = data.drop_duplicates(
        subset="ticker"
    ).reset_index(drop=True)

    return data


# ============================================================
# Download the entire universe
# ============================================================

def download_universe_prices(
    universe_path: str = DEFAULT_UNIVERSE_PATH,
    start: str = "2020-01-01",
    end: str = "2026-08-08",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    delay: float = 2.0,
) -> pd.DataFrame:
    """
    Download historical price data for every ticker
    in the universe.

    Existing valid files are skipped.

    Invalid or incomplete files are downloaded again.

    Parameters
    ----------
    universe_path : str
        Path to the universe CSV.

    start : str
        Historical start date.

    end : str
        Historical end date.

    output_dir : str
        Directory where price CSVs are stored.

    delay : float
        Number of seconds to wait between downloads.

    Returns
    -------
    pandas.DataFrame
        Download summary containing:

        ticker
        status
        error
    """

    # --------------------------------------------------------
    # Load universe
    # --------------------------------------------------------

    universe = load_universe(universe_path)

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(universe)

    results = []

    # --------------------------------------------------------
    # Download each ticker
    # --------------------------------------------------------

    for position, row in universe.iterrows():

        ticker = row["ticker"]

        file_path = output_path / f"{ticker}.csv"

        print(
            f"[{position + 1}/{total}] {ticker}"
        )

        # ----------------------------------------------------
        # Check whether a usable file already exists
        # ----------------------------------------------------

        if is_valid_price_file(file_path):

            print(
                f"  Skipping — valid file already exists: "
                f"{file_path}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "status": "skipped",
                    "error": None,
                }
            )

            continue

        # ----------------------------------------------------
        # Download data
        # ----------------------------------------------------

        try:

            data = download_price_data(
                ticker=ticker,
                start=start,
                end=end,
                output_dir=output_dir,
            )

            # ------------------------------------------------
            # Verify downloaded file
            # ------------------------------------------------

            if not is_valid_price_file(file_path):

                raise ValueError(
                    "Downloaded file failed validation."
                )

            print(
                f"  Downloaded {len(data)} rows"
            )

            results.append(
                {
                    "ticker": ticker,
                    "status": "success",
                    "error": None,
                }
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "status": "error",
                    "error": str(exc),
                }
            )

        # ----------------------------------------------------
        # Delay between requests
        # ----------------------------------------------------

        if position < total - 1:
            time.sleep(delay)

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = pd.DataFrame(results)

    summary_path = (
        output_path / "download_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("Download complete.")

    if not summary.empty:
        print(
            summary["status"]
            .value_counts()
        )

    print()
    print(
        f"Summary saved to: {summary_path}"
    )

    return summary


# ============================================================
# Command-line execution
# ============================================================

if __name__ == "__main__":

    summary = download_universe_prices()

    print()
    print(summary.to_string(index=False))