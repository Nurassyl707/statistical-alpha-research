from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SIGNALS_PATH = Path(
    "data/processed/cross_sectional_ranked.csv"
)

PRICES_DIR = Path(
    "data/raw/prices"
)

OUTPUT_PATH = Path(
    "data/processed/factor_backtest.csv"
)


# ============================================================
# Configuration
# ============================================================

TOP_QUANTILE = 0.10


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(returns):
    returns = (
        pd.Series(returns)
        .dropna()
    )

    if returns.empty:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Total Return": np.nan,
        }

    equity = (
        1 + returns
    ).cumprod()

    years = len(returns) / 252

    total_return = (
        equity.iloc[-1] - 1
    )

    cagr = (
        equity.iloc[-1]
        ** (1 / years)
        - 1
    )

    volatility = (
        returns.std()
        * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = (
        equity
        / equity.cummax()
        - 1
    )

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
        "Total Return": total_return,
    }


# ============================================================
# Load signals
# ============================================================

def load_signals():

    print("Loading signals...")

    df = pd.read_csv(
        SIGNALS_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    required = [
        "Date",
        "ticker",

        "return_21d_rank",
        "return_63d_rank",
        "return_126d_rank",
        "return_252d_rank",

        "volatility_21d_rank",
        "volatility_63d_rank",
        "volatility_252d_rank",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# Load prices
# ============================================================

def load_prices():

    print("Loading prices...")

    frames = []

    files = sorted(
        PRICES_DIR.glob("*.csv")
    )

    for file_path in files:

        if file_path.name == "download_summary.csv":
            continue

        ticker = file_path.stem

        data = pd.read_csv(
            file_path
        )

        if (
            "Date" not in data.columns
            or "Adj Close" not in data.columns
        ):
            continue

        data = data[
            [
                "Date",
                "Adj Close",
            ]
        ].copy()

        data["Date"] = pd.to_datetime(
            data["Date"]
        )

        data["ticker"] = ticker

        data = data.sort_values(
            "Date"
        )

        # Return from today's close
        # to next trading day's close.
        data["next_return"] = (
            data["Adj Close"].shift(-1)
            / data["Adj Close"]
            - 1
        )

        frames.append(data)

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
        subset=[
            "Date",
            "ticker",
        ]
    )

    print(
        f"Loaded {prices['ticker'].nunique()} stocks."
    )

    print(
        f"Price observations: "
        f"{len(prices):,}"
    )

    print(
        f"Price date range: "
        f"{prices['Date'].min().date()} "
        f"to "
        f"{prices['Date'].max().date()}"
    )

    return prices


# ============================================================
# Construct monthly portfolios
# ============================================================

def construct_portfolio(
    signals,
    signal_column,
    ascending=False,
):
    """
    Construct equal-weight monthly portfolio.

    ascending=False:
        highest factor ranks selected.

    ascending=True:
        lowest factor ranks selected.

    Portfolio is formed using information
    available at the month-end date.
    """

    data = signals.copy()

    data = data.dropna(
        subset=[signal_column]
    )

    # --------------------------------------------------------
    # Identify month-end signal date.
    # --------------------------------------------------------

    data["month"] = (
        data["Date"]
        .dt.to_period("M")
    )

    rebalance_dates = (
        data.groupby("month")["Date"]
        .max()
    )

    data = data[
        data["Date"].isin(
            rebalance_dates.values
        )
    ].copy()

    # --------------------------------------------------------
    # Cross-sectional ranking.
    # --------------------------------------------------------

    data["rank"] = (
        data.groupby("Date")[signal_column]
        .rank(
            method="first",
            ascending=ascending,
        )
    )

    data["n_stocks"] = (
        data.groupby("Date")["ticker"]
        .transform("count")
    )

    data["n_holdings"] = (
        data["n_stocks"]
        * TOP_QUANTILE
    ).astype(int)

    data["n_holdings"] = (
        data["n_holdings"]
        .clip(lower=1)
    )

    # --------------------------------------------------------
    # Select portfolio.
    # --------------------------------------------------------

    selected = data[
        data["rank"]
        <= data["n_holdings"]
    ].copy()

    # --------------------------------------------------------
    # Equal weights.
    # --------------------------------------------------------

    selected["weight"] = (
        1.0
        / selected.groupby("Date")
        ["ticker"]
        .transform("count")
    )

    return selected[
        [
            "Date",
            "ticker",
            "weight",
        ]
    ]


# ============================================================
# Convert monthly portfolio into daily returns
# ============================================================

def calculate_portfolio_returns(
    portfolio,
    prices,
    start_date,
    end_date,
):
    """
    Portfolio formed at month-end t.

    Returns begin on the NEXT available
    trading day after t.

    Positions remain unchanged until
    the next rebalance.
    """

    portfolio = portfolio.copy()

    portfolio_dates = sorted(
        portfolio["Date"].unique()
    )

    all_daily_returns = []

    for i, rebalance_date in enumerate(
        portfolio_dates
    ):

        rebalance_date = pd.Timestamp(
            rebalance_date
        )

        # ----------------------------------------------------
        # Determine next rebalance.
        # ----------------------------------------------------

        if i + 1 < len(
            portfolio_dates
        ):

            next_rebalance = pd.Timestamp(
                portfolio_dates[i + 1]
            )

        else:

            next_rebalance = (
                pd.Timestamp(end_date)
            )

        # ----------------------------------------------------
        # Portfolio holdings.
        # ----------------------------------------------------

        weights = portfolio[
            portfolio["Date"]
            == rebalance_date
        ][
            [
                "ticker",
                "weight",
            ]
        ].copy()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Start STRICTLY AFTER rebalance date.
        # ----------------------------------------------------

        period_prices = prices[
            (prices["Date"] > rebalance_date)
            & (
                prices["Date"]
                <= next_rebalance
            )
            & (
                prices["Date"]
                >= start_date
            )
            & (
                prices["Date"]
                <= end_date
            )
        ].copy()

        if period_prices.empty:
            continue

        # ----------------------------------------------------
        # Merge weights with daily returns.
        # ----------------------------------------------------

        period = period_prices.merge(
            weights,
            on="ticker",
            how="inner",
        )

        # ----------------------------------------------------
        # Weighted returns.
        # ----------------------------------------------------

        period["weighted_return"] = (
            period["next_return"]
            * period["weight"]
        )

        daily = (
            period.groupby("Date")
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

        all_daily_returns.append(
            daily
        )

    if not all_daily_returns:
        return pd.Series(
            dtype=float
        )

    result = pd.concat(
        all_daily_returns,
        ignore_index=True,
    )

    result = (
        result
        .drop_duplicates(
            subset=["Date"]
        )
        .sort_values("Date")
    )

    return result.set_index(
        "Date"
    )["portfolio_return"]


# ============================================================
# Main
# ============================================================

def main():

    signals = load_signals()

    prices = load_prices()

    # ========================================================
    # Create factor signals
    # ========================================================

    signals = signals.copy()

    # --------------------------------------------------------
    # Momentum composite
    # --------------------------------------------------------

    signals[
        "momentum_composite"
    ] = (
        0.10
        * signals["return_21d_rank"]
        + 0.20
        * signals["return_63d_rank"]
        + 0.30
        * signals["return_126d_rank"]
        + 0.40
        * signals["return_252d_rank"]
    )

    # --------------------------------------------------------
    # Low-volatility composite
    #
    # LOWER volatility is better.
    # --------------------------------------------------------

    signals[
        "low_volatility_composite"
    ] = (
        signals[
            [
                "volatility_21d_rank",
                "volatility_63d_rank",
                "volatility_252d_rank",
            ]
        ]
        .mean(axis=1)
    )

    # ========================================================
    # Strategy definitions
    # ========================================================

    strategies = {

        "21D Momentum": (
            "return_21d_rank",
            False,
        ),

        "63D Momentum": (
            "return_63d_rank",
            False,
        ),

        "126D Momentum": (
            "return_126d_rank",
            False,
        ),

        "252D Momentum": (
            "return_252d_rank",
            False,
        ),

        "Momentum Composite": (
            "momentum_composite",
            False,
        ),

        "Low Volatility": (
            "low_volatility_composite",
            True,
        ),
    }

    # ========================================================
    # Determine common evaluation period
    # ========================================================

    # The 252-day factor is the slowest factor.
    # Start only once its signal exists.

    valid_252 = signals.dropna(
        subset=[
            "return_252d_rank"
        ]
    )

    valid_dates = sorted(
        valid_252["Date"].unique()
    )

    if not valid_dates:
        raise ValueError(
            "No valid 252D factor dates."
        )

    first_signal_date = pd.Timestamp(
        valid_dates[0]
    )

    last_price_date = pd.Timestamp(
        prices["Date"].max()
    )

    print()
    print(
        "Common Evaluation Period"
    )
    print(
        "========================="
    )

    print(
        f"Start: "
        f"{first_signal_date.date()}"
    )

    print(
        f"End:   "
        f"{last_price_date.date()}"
    )

    # ========================================================
    # Run strategies
    # ========================================================

    results = []

    return_series = {}

    for name, (
        signal_column,
        ascending,
    ) in strategies.items():

        print()
        print(
            f"Running: {name}"
        )

        portfolio = construct_portfolio(
            signals,
            signal_column,
            ascending=ascending,
        )

        returns = calculate_portfolio_returns(
            portfolio=portfolio,
            prices=prices,
            start_date=first_signal_date,
            end_date=last_price_date,
        )

        # ----------------------------------------------------
        # Ensure common period.
        # ----------------------------------------------------

        returns = returns[
            (
                returns.index
                >= first_signal_date
            )
            & (
                returns.index
                <= last_price_date
            )
        ]

        return_series[name] = returns

    # ========================================================
    # Force identical dates
    # ========================================================

    combined = pd.concat(
        return_series,
        axis=1,
        join="inner",
    )

    combined = combined.dropna()

    print()
    print(
        "Common Daily Return Matrix"
    )
    print(
        "=========================="
    )

    print(
        f"Trading days: "
        f"{len(combined):,}"
    )

    print(
        f"Start: "
        f"{combined.index.min().date()}"
    )

    print(
        f"End: "
        f"{combined.index.max().date()}"
    )

    # ========================================================
    # Calculate metrics
    # ========================================================

    for name in strategies:

        returns = combined[name]

        metrics = calculate_metrics(
            returns
        )

        metrics["Strategy"] = name

        metrics["Trading Days"] = (
            len(returns)
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Strategy",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
            "Total Return",
            "Trading Days",
        ]
    ]

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Final Factor Backtest"
    )
    print(
        "====================="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
