from pathlib import Path
import time

import pandas as pd
import yfinance as yf


EXPECTED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


def download_price_data(
    ticker: str,
    start: str,
    end: str,
    output_dir: str = "data/raw/prices",
    max_retries: int = 3,
    retry_delay: int = 5,
) -> pd.DataFrame:
    """
    Download historical daily price data for one ticker.

    Data is normalized into:
        Date, Open, High, Low, Close, Adj Close, Volume

    Existing local files are reused when possible.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{ticker}.csv"

    # ---------------------------------------------------------
    # Cache: avoid repeatedly requesting the same ticker.
    # ---------------------------------------------------------
    if file_path.exists():
        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError(f"Cached file is empty: {file_path}")

        data["Date"] = pd.to_datetime(data["Date"])

        return data

    # ---------------------------------------------------------
    # Download with retry/backoff.
    # ---------------------------------------------------------
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Downloading {ticker} "
                f"(attempt {attempt}/{max_retries})..."
            )

            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                raise ValueError(
                    f"No data returned for ticker {ticker}."
                )

            break

        except Exception as exc:
            last_error = exc

            if attempt < max_retries:
                wait_time = retry_delay * attempt

                print(
                    f"Download failed for {ticker}: {exc}"
                )
                print(
                    f"Waiting {wait_time} seconds before retry..."
                )

                time.sleep(wait_time)

            else:
                raise RuntimeError(
                    f"Failed to download {ticker} "
                    f"after {max_retries} attempts."
                ) from last_error

    # ---------------------------------------------------------
    # Normalize yfinance output.
    # ---------------------------------------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns for {ticker}: "
            f"{missing_columns}"
        )

    data = data[EXPECTED_COLUMNS]

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Save locally.
    # ---------------------------------------------------------
    data.to_csv(file_path, index=False)

    print(
        f"Saved {ticker}: "
        f"{len(data):,} rows -> {file_path}"
    )

    return data


def load_price_data(
    ticker: str,
    data_dir: str = "data/raw/prices",
) -> pd.DataFrame:
    """
    Load previously downloaded price data.
    """

    file_path = Path(data_dir) / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Price data not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(data["Date"])

    return data