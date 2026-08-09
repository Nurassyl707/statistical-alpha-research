from io import StringIO
from pathlib import Path

import pandas as pd
import requests


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def download_sp500_universe(
    output_path="data/raw/universe/sp500.csv",
):
    """
    Download the current S&P 500 constituent list from Wikipedia.

    Parameters
    ----------
    output_path : str
        Location where the universe CSV will be saved.

    Returns
    -------
    pd.DataFrame
        S&P 500 constituent data.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0 Safari/537.36"
        )
    }

    response = requests.get(
        SP500_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise ValueError("No tables found on the S&P 500 Wikipedia page.")

    universe = tables[0].copy()

    # Rename the important columns to stable names.
    universe = universe.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )

    # Yahoo Finance uses '-' instead of '.' for some tickers.
    universe["ticker"] = universe["ticker"].str.replace(
        ".", "-", regex=False
    )

    # Keep only the columns we need for the research pipeline.
    columns = [
        "ticker",
        "company",
        "sector",
        "sub_industry",
    ]

    universe = universe[columns]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    universe.to_csv(output, index=False)

    return universe


def load_sp500_universe(
    path="data/raw/universe/sp500.csv",
):
    """
    Load a previously downloaded S&P 500 universe.
    """

    return pd.read_csv(path)
