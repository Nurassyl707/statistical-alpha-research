from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/alpha_signals.csv")
OUTPUT_PATH = Path("data/processed/portfolio_weights.csv")


def build_monthly_portfolio(
    input_path: str = str(INPUT_PATH),
    output_path: str = str(OUTPUT_PATH),
    top_quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Construct a monthly-rebalanced, equal-weighted long-only portfolio.

    Strategy:
        1. Load alpha signals.
        2. Select the last available trading day of each month.
        3. Select the top `top_quantile` of stocks by alpha rank.
        4. Equal-weight selected stocks.
        5. Save portfolio weights.

    Parameters
    ----------
    input_path : str
        Path to alpha signal dataset.

    output_path : str
        Path where portfolio weights will be saved.

    top_quantile : float
        Fraction of stocks to hold.
        0.10 means top 10%.

    Returns
    -------
    pandas.DataFrame
        Portfolio holdings and weights.
    """

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])

    required_columns = [
        "Date",
        "ticker",
        "alpha_rank",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if not 0 < top_quantile <= 1:
        raise ValueError(
            "top_quantile must be between 0 and 1."
        )

    # ---------------------------------------------------------
    # Keep only observations with a valid alpha signal.
    # ---------------------------------------------------------

    df = df.dropna(
        subset=["alpha_rank"]
    ).copy()

    # ---------------------------------------------------------
    # Identify the last trading day of each month.
    # ---------------------------------------------------------

    df["month"] = df["Date"].dt.to_period("M")

    rebalance_dates = (
        df.groupby("month")["Date"]
        .max()
        .reset_index(drop=True)
    )

    df = df[
        df["Date"].isin(rebalance_dates)
    ].copy()

    # ---------------------------------------------------------
    # Rank stocks within each rebalance date.
    # ---------------------------------------------------------

    df["cross_sectional_rank"] = (
        df.groupby("Date")["alpha_rank"]
        .rank(
            method="first",
            ascending=False,
        )
    )

    # Number of stocks available on each date.
    df["n_stocks"] = (
        df.groupby("Date")["ticker"]
        .transform("count")
    )

    # Number of stocks to hold.
    df["n_holdings"] = (
        df["n_stocks"] * top_quantile
    ).apply(lambda x: max(1, int(x)))
    
    # ---------------------------------------------------------
    # Select top-decile stocks.
    # ---------------------------------------------------------

    portfolio = df[
        df["cross_sectional_rank"]
        <= df["n_holdings"]
    ].copy()

    # ---------------------------------------------------------
    # Equal-weight portfolio.
    # ---------------------------------------------------------

    portfolio["weight"] = (
        1.0
        / portfolio.groupby("Date")["ticker"]
        .transform("count")
    )

    # ---------------------------------------------------------
    # Keep useful columns.
    # ---------------------------------------------------------

    portfolio = portfolio[
        [
            "Date",
            "ticker",
            "alpha_rank",
            "weight",
            "n_stocks",
            "n_holdings",
        ]
    ].sort_values(
        ["Date", "weight", "ticker"],
        ascending=[True, False, True],
    )

    portfolio = portfolio.reset_index(drop=True)

    # ---------------------------------------------------------
    # Validate portfolio weights.
    # ---------------------------------------------------------

    weight_check = (
        portfolio.groupby("Date")["weight"]
        .sum()
    )

    if not ((weight_check - 1.0).abs() < 1e-10).all():
        raise ValueError(
            "Portfolio weights do not sum to 1."
        )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    portfolio.to_csv(
        output,
        index=False,
    )

    print(
        "\nPortfolio construction complete."
    )

    print(
        f"Rebalance dates: "
        f"{portfolio['Date'].nunique():,}"
    )

    print(
        f"Unique stocks: "
        f"{portfolio['ticker'].nunique():,}"
    )

    print(
        f"Portfolio observations: "
        f"{len(portfolio):,}"
    )

    print(
        f"Date range: "
        f"{portfolio['Date'].min().date()} "
        f"to "
        f"{portfolio['Date'].max().date()}"
    )

    print(
        f"Average holdings: "
        f"{portfolio.groupby('Date')['ticker'].count().mean():.1f}"
    )

    print(
        f"Saved to: {output}"
    )

    return portfolio


def validate_portfolio(
    input_path: str = str(OUTPUT_PATH),
) -> None:
    """
    Validate portfolio construction.
    """

    portfolio = pd.read_csv(
        input_path
    )

    portfolio["Date"] = pd.to_datetime(
        portfolio["Date"]
    )

    print("\nPortfolio validation")
    print("====================")

    weight_sum = (
        portfolio.groupby("Date")["weight"]
        .sum()
    )

    print(
        f"Weight sum min: "
        f"{weight_sum.min():.6f}"
    )

    print(
        f"Weight sum max: "
        f"{weight_sum.max():.6f}"
    )

    print(
        f"Rebalance dates: "
        f"{portfolio['Date'].nunique():,}"
    )

    print(
        f"Average holdings: "
        f"{portfolio.groupby('Date')['ticker'].count().mean():.2f}"
    )

    print(
        f"Minimum holdings: "
        f"{portfolio.groupby('Date')['ticker'].count().min()}"
    )

    print(
        f"Maximum holdings: "
        f"{portfolio.groupby('Date')['ticker'].count().max()}"
    )

    print("\nLatest portfolio:")

    latest_date = portfolio["Date"].max()

    latest = portfolio[
        portfolio["Date"] == latest_date
    ].sort_values(
        "weight",
        ascending=False,
    )

    print(
        latest.to_string(index=False)
    )


if __name__ == "__main__":
    build_monthly_portfolio()
    validate_portfolio()