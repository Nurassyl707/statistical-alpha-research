from pathlib import Path

import numpy as np
import pandas as pd


PRICE_DIR = Path("data/raw/prices")
FEATURE_DIR = Path("data/processed/features")


def load_price_data(ticker: str) -> pd.DataFrame:
    """
    Load historical price data for one ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    pd.DataFrame
        Clean historical price data.
    """

    file_path = PRICE_DIR / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Price file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{ticker}: missing columns {missing}"
        )

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")
    data = data.drop_duplicates("Date")
    data = data.reset_index(drop=True)

    return data


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum and volatility features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing price and return features.
    """

    data = data.copy()

    # Use adjusted close for return calculations.
    price = data["Adj Close"]

    # Daily return.
    data["return_1d"] = price.pct_change(1)

    # Multi-period returns.
    data["return_5d"] = price.pct_change(5)
    data["return_21d"] = price.pct_change(21)
    data["return_63d"] = price.pct_change(63)
    data["return_126d"] = price.pct_change(126)
    data["return_252d"] = price.pct_change(252)

    # Rolling annualized volatility.
    data["volatility_21d"] = (
        data["return_1d"]
        .rolling(21)
        .std()
        * np.sqrt(252)
    )

    data["volatility_63d"] = (
        data["return_1d"]
        .rolling(63)
        .std()
        * np.sqrt(252)
    )

    data["volatility_252d"] = (
        data["return_1d"]
        .rolling(252)
        .std()
        * np.sqrt(252)
    )

    return data


def build_ticker_features(
    ticker: str,
    output_dir: str = "data/processed/features",
) -> pd.DataFrame:
    """
    Load price data, calculate features, and save the result.
    """

    data = load_price_data(ticker)

    features = calculate_returns(data)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{ticker}.csv"

    features.to_csv(file_path, index=False)

    print(
        f"Saved {ticker}: "
        f"{len(features):,} rows -> {file_path}"
    )

    return features


def build_all_features(
    universe_file: str = "data/raw/universe/sp500.csv",
    output_dir: str = "data/processed/features",
) -> pd.DataFrame:
    """
    Build return and volatility features for the entire universe.
    """

    universe = pd.read_csv(universe_file)

    results = []

    total = len(universe)

    for i, ticker in enumerate(universe["ticker"], start=1):

        print(f"[{i}/{total}] {ticker}")

        try:
            features = build_ticker_features(
                ticker=ticker,
                output_dir=output_dir,
            )

            results.append(
                {
                    "ticker": ticker,
                    "status": "success",
                    "rows": len(features),
                    "error": None,
                }
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            results.append(
                {
                    "ticker": ticker,
                    "status": "error",
                    "rows": 0,
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(results)

    summary_path = Path(output_dir) / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFeature engineering complete.")
    print(summary["status"].value_counts())
    print(f"\nSummary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    build_all_features()