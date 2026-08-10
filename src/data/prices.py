from pathlib import Path
import time

import pandas as pd
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

DEFAULT_OUTPUT_DIR = "data/raw/prices"

MAX_RETRIES = 3

RETRY_DELAY = 5


# ============================================================
# Download historical price data
# ============================================================

def download_price_data(
    ticker: str,
    start: str,
    end: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Download historical daily price data for one ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".

    start : str
        Start date in YYYY-MM-DD format.

    end : str
        End date in YYYY-MM-DD format.

    output_dir : str
        Directory where the raw CSV will be saved.

    Returns
    -------
    pandas.DataFrame
        Normalized historical market data.
    """

    ticker = ticker.strip().upper()

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = output_path / f"{ticker}.csv"

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"Downloading {ticker} "
            f"(attempt {attempt}/{MAX_RETRIES})..."
        )

        try:

            # ------------------------------------------------
            # Download using yfinance
            # ------------------------------------------------

            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            # ------------------------------------------------
            # Check whether data was returned
            # ------------------------------------------------

            if data is None or data.empty:

                raise ValueError(
                    f"No data returned for ticker {ticker}."
                )

            # ------------------------------------------------
            # Flatten MultiIndex columns
            # ------------------------------------------------

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):

                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            # ------------------------------------------------
            # Reset index
            # ------------------------------------------------

            data = data.reset_index()

            # ------------------------------------------------
            # Required columns
            # ------------------------------------------------

            expected_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]

            missing_columns = [
                column
                for column in expected_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing expected columns: "
                    f"{missing_columns}"
                )

            # ------------------------------------------------
            # Keep only required columns
            # ------------------------------------------------

            data = data[
                expected_columns
            ].copy()

            # ------------------------------------------------
            # Normalize dates
            # ------------------------------------------------

            data["Date"] = pd.to_datetime(
                data["Date"],
                errors="coerce",
            )

            # Remove invalid dates
            data = data.dropna(
                subset=["Date"]
            )

            # ------------------------------------------------
            # Convert numeric columns
            # ------------------------------------------------

            numeric_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]

            for column in numeric_columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

            # ------------------------------------------------
            # Remove rows with missing prices
            # ------------------------------------------------

            data = data.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            )

            # ------------------------------------------------
            # Sort chronologically
            # ------------------------------------------------

            data = data.sort_values(
                "Date"
            )

            # ------------------------------------------------
            # Remove duplicate dates
            # ------------------------------------------------

            data = data.drop_duplicates(
                subset="Date"
            )

            # ------------------------------------------------
            # Reset index
            # ------------------------------------------------

            data = data.reset_index(
                drop=True
            )

            # ------------------------------------------------
            # Final check
            # ------------------------------------------------

            if data.empty:

                raise ValueError(
                    f"No usable rows remain "
                    f"after cleaning {ticker}."
                )

            # ------------------------------------------------
            # Save CSV
            # ------------------------------------------------

            data.to_csv(
                file_path,
                index=False,
            )

            print(
                f"Saved {ticker}: "
                f"{len(data):,} rows -> "
                f"{file_path}"
            )

            return data

        except Exception as exc:

            print(
                f"  Download failed: {exc}"
            )

            if attempt < MAX_RETRIES:

                print(
                    f"  Retrying in "
                    f"{RETRY_DELAY} seconds..."
                )

                time.sleep(
                    RETRY_DELAY
                )

            else:

                raise


# ============================================================
# Load price data
# ============================================================

def load_price_data(
    ticker: str,
    input_dir: str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Load previously downloaded price data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    input_dir : str
        Directory containing price CSV files.

    Returns
    -------
    pandas.DataFrame
        Historical price data.
    """

    ticker = ticker.strip().upper()

    file_path = (
        Path(input_dir)
        / f"{ticker}.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Price file not found: {file_path}"
        )

    data = pd.read_csv(
        file_path
    )

    if "Date" not in data.columns:

        raise ValueError(
            f"Missing Date column in "
            f"{file_path}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Date"]
    )

    data = data.sort_values(
        "Date"
    )

    data = data.drop_duplicates(
        subset="Date"
    )

    data = data.reset_index(
        drop=True
    )

    return data