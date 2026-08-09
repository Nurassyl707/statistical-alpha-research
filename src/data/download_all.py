from pathlib import Path
import time

import pandas as pd

from src.data.prices import download_price_data


def download_universe_prices(
    universe_path: str = "data/raw/universe/sp500.csv",
    start: str = "2020-01-01",
    end: str = "2026-08-08",
    output_dir: str = "data/raw/prices",
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Download historical price data for every ticker in the S&P 500 universe.

    Existing files are skipped so the function can be safely re-run.

    Returns
    -------
    pandas.DataFrame
        Download summary with ticker, status, and error information.
    """

    universe = pd.read_csv(universe_path)

    if "ticker" not in universe.columns:
        raise ValueError(
            f"Universe file must contain a 'ticker' column. "
            f"Found: {universe.columns.tolist()}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

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

    for i, ticker in enumerate(tickers, start=1):
        file_path = output_path / f"{ticker}.csv"

        print(f"[{i}/{total}] {ticker}")

        # Skip data that already exists.
        if file_path.exists():
            print(f"  Skipping — already exists: {file_path}")

            results.append(
                {
                    "ticker": ticker,
                    "status": "skipped",
                    "error": None,
                }
            )

            continue

        try:
            df = download_price_data(
                ticker=ticker,
                start=start,
                end=end,
                output_dir=output_dir,
            )

            print(f"  Downloaded {len(df)} rows")

            results.append(
                {
                    "ticker": ticker,
                    "status": "success",
                    "error": None,
                }
            )

        except Exception as exc:
            print(f"  FAILED: {exc}")

            results.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        time.sleep(delay)

    summary = pd.DataFrame(results)

    summary_path = output_path / "download_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nDownload complete.")
    print(summary["status"].value_counts())

    return summary