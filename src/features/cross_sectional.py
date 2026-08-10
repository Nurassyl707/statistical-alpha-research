from pathlib import Path

import pandas as pd


FEATURE_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("data/processed")


FACTOR_COLUMNS = [
    "return_21d",
    "return_63d",
    "return_126d",
    "return_252d",
    "volatility_21d",
    "volatility_63d",
]


def load_feature_data(ticker: str) -> pd.DataFrame:
    """
    Load engineered features for one ticker.
    """

    file_path = FEATURE_DIR / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(data["Date"])

    data["ticker"] = ticker

    return data


def build_cross_sectional_dataset(
    universe_file: str = "data/raw/universe/sp500.csv",
    output_file: str = "data/processed/cross_sectional.csv",
) -> pd.DataFrame:
    """
    Combine all stock-level feature files into one
    cross-sectional panel.

    Each row represents:

        date + ticker

    This creates the dataset required for cross-sectional
    factor ranking and portfolio construction.
    """

    universe = pd.read_csv(universe_file)

    all_data = []

    total = len(universe)

    for i, ticker in enumerate(universe["ticker"], start=1):

        print(f"[{i}/{total}] Loading {ticker}")

        try:
            data = load_feature_data(ticker)

            columns = [
                "Date",
                "ticker",
                "Adj Close",
                "return_1d",
                "return_5d",
                "return_21d",
                "return_63d",
                "return_126d",
                "return_252d",
                "volatility_21d",
                "volatility_63d",
                "volatility_252d",
            ]

            data = data[columns]

            all_data.append(data)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not all_data:
        raise ValueError("No feature data loaded.")

    panel = pd.concat(
        all_data,
        ignore_index=True,
    )

    panel = panel.sort_values(
        ["Date", "ticker"]
    )

    panel = panel.drop_duplicates(
        subset=["Date", "ticker"]
    )

    panel = panel.reset_index(drop=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_csv(
        output_path,
        index=False,
    )

    print("\nCross-sectional dataset created.")
    print(f"Rows: {len(panel):,}")
    print(f"Stocks: {panel['ticker'].nunique()}")
    print(
        f"Dates: "
        f"{panel['Date'].min().date()} "
        f"to "
        f"{panel['Date'].max().date()}"
    )
    print(f"Saved to: {output_path}")

    return panel


def add_cross_sectional_ranks(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add daily cross-sectional percentile ranks.

    Ranking is performed independently for each date.
    """

    panel = panel.copy()

    momentum_columns = [
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
    ]

    volatility_columns = [
        "volatility_21d",
        "volatility_63d",
        "volatility_252d",
    ]

    for column in momentum_columns:

        panel[f"{column}_rank"] = (
            panel.groupby("Date")[column]
            .rank(
                pct=True,
                method="average",
            )
        )

    for column in volatility_columns:

        panel[f"{column}_rank"] = (
            panel.groupby("Date")[column]
            .rank(
                pct=True,
                method="average",
            )
        )

    return panel


if __name__ == "__main__":

    panel = build_cross_sectional_dataset()

    panel = add_cross_sectional_ranks(panel)

    output_file = (
        OUTPUT_DIR /
        "cross_sectional_ranked.csv"
    )

    panel.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nRanked dataset saved to: {output_file}"
    )