from pathlib import Path
import time

import pandas as pd

from src.data.prices import download_price_data


def is_valid_price_file(file_path: Path) -> bool:
    """
    Check whether an existing price CSV has the expected structure
    and contains usable market data.
    """

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    try:
        df = pd.read_csv(file_path)

        # Check that all expected columns exist
        if df.columns.tolist() != required_columns:
            return False

        # File must contain data
        if df.empty:
            return False

        # Dates must be valid
        dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

        if dates.isna().any():
            return False

        # Required market columns must contain numeric values
        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]

        for column in numeric_columns:
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if values.isna().any():
                return False

        # Dates should not be duplicated
        if dates.duplicated().any():
            return False

        return True

    except Exception:
        return False


def download_universe_prices(
    universe_path: str = "data/raw/universe/sp500.csv",
    start: str = "2020-01-01",
    end: str = "2026-08-08",
    output_dir: str = "data/raw/prices",
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Download historical price data for every ticker in a stock universe.

    Existing valid files are skipped.
    Existing invalid files are downloaded again.

    Parameters
    ----------
    universe_path : str
        Path to CSV containing a 'ticker' column.

    start : str
        Start date in YYYY-MM-DD format.

    end : str
        End date in YYYY-MM-DD format.

    output_dir : str
        Directory where price CSV files are stored.

    delay : float
        Number of seconds to wait between downloads.

    Returns
    -------
    pandas.DataFrame
        Download summary.
    """

    # Load universe
    universe = pd.read_csv(universe_path)

    # Validate universe structure
    if "ticker" not in universe.columns:
        raise ValueError(
            "Universe file must contain a 'ticker' column. "
            f"Found columns: {universe.columns.tolist()}"
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clean ticker list
    tickers = (
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )

    total = len(tickers)

    results = []

    # Process each ticker
    for i, ticker in enumerate(tickers, start=1):

        print(f"[{i}/{total}] {ticker}")

        file_path = output_path / f"{ticker}.csv"

        # ---------------------------------------------------------
        # Check whether an existing file is valid
        # ---------------------------------------------------------
        if file_path.exists():

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

            print(
                "  Existing file is invalid — "
                "redownloading."
            )

        # ---------------------------------------------------------
        # Download price data
        # ---------------------------------------------------------
        try:

            df = download_price_data(
                ticker=ticker,
                start=start,
                end=end,
                output_dir=output_dir,
            )

            print(
                f"  Downloaded {len(df)} rows"
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
                f"  FAILED: {exc}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        # ---------------------------------------------------------
        # Delay between requests
        # ---------------------------------------------------------
        time.sleep(delay)

    # -------------------------------------------------------------
    # Create download summary
    # -------------------------------------------------------------

    summary = pd.DataFrame(results)

    summary_path = (
        output_path / "download_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nDownload complete.")

    if not summary.empty:
        print(
            summary["status"].value_counts()
        )

    print(
        f"\nSummary saved to: {summary_path}"
    )

    return summary