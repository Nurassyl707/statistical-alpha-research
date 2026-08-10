from pathlib import Path

import pandas as pd
import numpy as np


PRICES_DIR = Path("data/raw/prices")
PORTFOLIO_PATH = Path("data/processed/portfolio_weights.csv")

OUTPUT_PATH = Path(
    "data/processed/backtest_returns.csv"
)


def load_price_data(
    prices_dir: str = str(PRICES_DIR),
) -> pd.DataFrame:
    """
    Load all historical price files into one DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns:
            Date
            ticker
            Adj Close
    """

    prices_path = Path(prices_dir)

    files = sorted(
        prices_path.glob("*.csv")
    )

    files = [
        file
        for file in files
        if file.name != "download_summary.csv"
    ]

    if not files:
        raise FileNotFoundError(
            f"No price files found in {prices_path}"
        )

    frames = []

    print(
        f"Loading {len(files)} price files..."
    )

    for i, file_path in enumerate(
        files,
        start=1,
    ):
        ticker = file_path.stem

        print(
            f"[{i}/{len(files)}] Loading {ticker}"
        )

        df = pd.read_csv(
            file_path
        )

        required = [
            "Date",
            "Adj Close",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            print(
                f"  Skipping {ticker}: "
                f"missing {missing}"
            )
            continue

        df = df[
            ["Date", "Adj Close"]
        ].copy()

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df["ticker"] = ticker

        frames.append(df)

    if not frames:
        raise ValueError(
            "No valid price files found."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices = prices.sort_values(
        ["Date", "ticker"]
    )

    prices = prices.drop_duplicates(
        subset=["Date", "ticker"]
    )

    prices = prices.reset_index(
        drop=True
    )

    return prices


def load_portfolio(
    portfolio_path: str = str(
        PORTFOLIO_PATH
    ),
) -> pd.DataFrame:
    """
    Load portfolio weights.
    """

    portfolio = pd.read_csv(
        portfolio_path
    )

    portfolio["Date"] = pd.to_datetime(
        portfolio["Date"]
    )

    required = [
        "Date",
        "ticker",
        "weight",
    ]

    missing = [
        column
        for column in required
        if column not in portfolio.columns
    ]

    if missing:
        raise ValueError(
            f"Missing portfolio columns: "
            f"{missing}"
        )

    return portfolio[
        required
    ].copy()


def build_daily_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily returns from adjusted close prices.
    """

    prices = prices.sort_values(
        ["ticker", "Date"]
    ).copy()

    prices["daily_return"] = (
        prices
        .groupby("ticker")["Adj Close"]
        .pct_change()
    )

    return prices


def run_backtest(
    portfolio_path: str = str(
        PORTFOLIO_PATH
    ),
    prices_dir: str = str(
        PRICES_DIR
    ),
    output_path: str = str(
        OUTPUT_PATH
    ),
) -> pd.DataFrame:
    """
    Run a monthly-rebalanced portfolio backtest.

    Important:
        Portfolio weights generated on date t are
        applied to returns AFTER date t.

    This prevents look-ahead bias.
    """

    print(
        "\nLoading portfolio..."
    )

    portfolio = load_portfolio(
        portfolio_path
    )

    print(
        f"Portfolio rows: "
        f"{len(portfolio):,}"
    )

    print(
        "\nLoading prices..."
    )

    prices = load_price_data(
        prices_dir
    )

    print(
        f"Price rows: "
        f"{len(prices):,}"
    )

    # ---------------------------------------------------------
    # Calculate daily returns.
    # ---------------------------------------------------------

    prices = build_daily_returns(
        prices
    )

    # ---------------------------------------------------------
    # Determine the next rebalance date.
    # ---------------------------------------------------------

    rebalance_dates = sorted(
        portfolio["Date"].unique()
    )

    all_dates = sorted(
        prices["Date"].unique()
    )

    portfolio_returns = []

    print(
        "\nRunning backtest..."
    )

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):
        rebalance_date = pd.Timestamp(
            rebalance_date
        )

        # -----------------------------------------------------
        # Portfolio remains active until the next rebalance.
        # -----------------------------------------------------

        if i + 1 < len(
            rebalance_dates
        ):
            next_rebalance = pd.Timestamp(
                rebalance_dates[i + 1]
            )

            period_end = (
                next_rebalance
            )

        else:
            period_end = pd.Timestamp(
                max(all_dates)
            )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Start AFTER the rebalance date.
        # -----------------------------------------------------

        period_prices = prices[
            (prices["Date"] > rebalance_date)
            & (
                prices["Date"]
                <= period_end
            )
        ].copy()

        if period_prices.empty:
            continue

        # -----------------------------------------------------
        # Get portfolio weights.
        # -----------------------------------------------------

        weights = portfolio[
            portfolio["Date"]
            == rebalance_date
        ][
            ["ticker", "weight"]
        ].copy()

        # -----------------------------------------------------
        # Merge weights with daily returns.
        # -----------------------------------------------------

        period = period_prices.merge(
            weights,
            on="ticker",
            how="inner",
        )

        # -----------------------------------------------------
        # Weighted portfolio return.
        # -----------------------------------------------------

        period["weighted_return"] = (
            period["daily_return"]
            * period["weight"]
        )

        daily = (
            period
            .groupby("Date")
            ["weighted_return"]
            .sum(
                min_count=1
            )
            .reset_index()
        )

        daily = daily.rename(
            columns={
                "weighted_return":
                    "portfolio_return"
            }
        )

        daily[
            "rebalance_date"
        ] = rebalance_date

        portfolio_returns.append(
            daily
        )

    if not portfolio_returns:
        raise ValueError(
            "Backtest produced no returns."
        )

    result = pd.concat(
        portfolio_returns,
        ignore_index=True,
    )

    result = result.sort_values(
        "Date"
    )

    result = result.drop_duplicates(
        subset=["Date"]
    )

    result = result.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Cumulative wealth.
    # ---------------------------------------------------------

    result[
        "cumulative_return"
    ] = (
        1.0
        + result["portfolio_return"]
    ).cumprod()

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    print(
        "\nBacktest complete."
    )

    print(
        f"Trading days: "
        f"{len(result):,}"
    )

    print(
        f"Date range: "
        f"{result['Date'].min().date()} "
        f"to "
        f"{result['Date'].max().date()}"
    )

    print(
        f"Mean daily return: "
        f"{result['portfolio_return'].mean():.6f}"
    )

    print(
        f"Daily volatility: "
        f"{result['portfolio_return'].std():.6f}"
    )

    print(
        f"Final cumulative wealth: "
        f"{result['cumulative_return'].iloc[-1]:.4f}"
    )

    print(
        f"Saved to: {output}"
    )

    return result


def validate_backtest(
    input_path: str = str(
        OUTPUT_PATH
    ),
) -> None:
    """
    Validate backtest output.
    """

    df = pd.read_csv(
        input_path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    print(
        "\nBacktest validation"
    )

    print(
        "==================="
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Missing daily returns: "
        f"{df['portfolio_return'].isna().sum():,}"
    )

    print(
        f"Mean daily return: "
        f"{df['portfolio_return'].mean():.8f}"
    )

    print(
        f"Std daily return: "
        f"{df['portfolio_return'].std():.8f}"
    )

    print(
        f"Minimum daily return: "
        f"{df['portfolio_return'].min():.6f}"
    )

    print(
        f"Maximum daily return: "
        f"{df['portfolio_return'].max():.6f}"
    )

    print(
        f"Final cumulative wealth: "
        f"{df['cumulative_return'].iloc[-1]:.6f}"
    )

    print(
        "\nLast 10 observations:"
    )

    print(
        df.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run_backtest()
    validate_backtest()