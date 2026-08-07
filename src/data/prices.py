from pathlib import Path

import pandas as pd
import yfinance as yf


def download_price_data(
    ticker: str,
    start: str,
    end: str,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Download historical daily price data for one ticker.

    The raw data is normalized into a flat DataFrame with:
        Date, Open, High, Low, Close, Adj Close, Volume

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    output_dir : str
        Directory where raw data will be saved.

    Returns
    -------
    pandas.DataFrame
        Normalized historical market data.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned for ticker {ticker}."
        )

    # yfinance may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

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
            f"Missing expected columns: {missing_columns}"
        )

    data = data[expected_columns]

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")
    data = data.drop_duplicates(subset="Date")
    data = data.reset_index(drop=True)

    file_path = output_path / f"{ticker}.csv"
    data.to_csv(file_path, index=False)

    return data