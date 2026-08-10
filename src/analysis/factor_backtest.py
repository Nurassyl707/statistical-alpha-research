from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS_PATH = Path(
    "data/processed/cross_sectional_ranked.csv"
)

PRICES_DIR = Path(
    "data/raw/prices"
)

OUTPUT_PATH = Path(
    "data/processed/factor_backtest.csv"
)


def calculate_metrics(returns):
    returns = pd.Series(returns).dropna()

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
        equity.iloc[-1] ** (1 / years) - 1
    )

    volatility = (
        returns.std() * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = (
        equity / equity.cummax()
        - 1
    )

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
        "Total Return": total_return,
    }


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

    return prices


def construct_portfolio(
    signals,
    signal_column,
    top_quantile=0.10,
):
    data = signals.copy()

    data = data.dropna(
        subset=[signal_column]
    )

    # Last available trading day
    # of each calendar month.
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

    # Cross-sectional ranking.
    data["rank"] = (
        data.groupby("Date")[signal_column]
        .rank(
            method="first",
            ascending=False,
        )
    )

    data["n_stocks"] = (
        data.groupby("Date")["ticker"]
        .transform("count")
    )

    data["n_holdings"] = (
        data["n_stocks"]
        * top_quantile
    ).apply(
        lambda x: max(1, int(x))
    )

    selected = data[
        data["rank"]
        <= data["n_holdings"]
    ].copy()

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


def calculate_portfolio_returns(
    portfolio,
    prices,
):
    data = portfolio.merge(
        prices[
            [
                "Date",
                "ticker",
                "next_return",
            ]
        ],
        on=[
            "Date",
            "ticker",
        ],
        how="left",
    )

    data["contribution"] = (
        data["weight"]
        * data["next_return"]
    )

    # IMPORTANT:
    #
    # The return associated with a
    # rebalance date is the NEXT
    # trading day's return.
    #
    # Therefore the portfolio is
    # formed using information available
    # at the rebalance date and does
    # not use future returns.

    returns = (
        data.groupby("Date")
        ["contribution"]
        .sum(min_count=1)
    )

    return returns


def main():

    signals = load_signals()

    prices = load_prices()

    strategies = {
        "21D Momentum": "return_21d_rank",

        "63D Momentum": "return_63d_rank",

        "126D Momentum": "return_126d_rank",

        "252D Momentum": "return_252d_rank",

        "Momentum Composite": None,

        "Low Volatility": None,
    }

    # Composite signals are constructed
    # entirely from ranks available on
    # the rebalance date.

    signals = signals.copy()

    signals["momentum_composite"] = (
        0.10
        * signals["return_21d_rank"]
        + 0.20
        * signals["return_63d_rank"]
        + 0.30
        * signals["return_126d_rank"]
        + 0.40
        * signals["return_252d_rank"]
    )

    signals["low_volatility_composite"] = (
        (
            signals[
                "volatility_21d_rank"
            ]
            + signals[
                "volatility_63d_rank"
            ]
            + signals[
                "volatility_252d_rank"
            ]
        )
        / 3
    )

    strategies[
        "Momentum Composite"
    ] = "momentum_composite"

    strategies[
        "Low Volatility"
    ] = "low_volatility_composite"

    results = []

    for name, signal_column in strategies.items():

        print(
            f"Running: {name}"
        )

        portfolio = construct_portfolio(
            signals,
            signal_column,
        )

        returns = calculate_portfolio_returns(
            portfolio,
            prices,
        )

        metrics = calculate_metrics(
            returns
        )

        metrics["Strategy"] = name

        metrics["Trading Days"] = (
            len(returns)
        )

        metrics["Rebalance Dates"] = (
            portfolio["Date"].nunique()
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
            "Rebalance Dates",
        ]
    ]

    print()
    print(
        "Clean Factor Backtest"
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

    output = OUTPUT_PATH

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Saved to: {output}"
    )


if __name__ == "__main__":
    main()