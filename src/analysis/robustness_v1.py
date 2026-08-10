import pandas as pd
import numpy as np
from pathlib import Path


SIGNALS_PATH = Path("data/processed/alpha_signals.csv")
OUTPUT_PATH = Path("data/processed/robustness_results.csv")


def load_data():
    df = pd.read_csv(SIGNALS_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["ticker", "Date"]).copy()

    df["next_return"] = (
        df.groupby("ticker")["Adj Close"].shift(-1)
        / df["Adj Close"]
        - 1
    )

    # Robustness test: exclude SNDK
    df = df[df["ticker"] != "SNDK"].copy()

    return df


def calculate_metrics(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Total Return": np.nan,
        }

    equity = (1 + returns).cumprod()

    years = len(returns) / 252

    total_return = equity.iloc[-1] - 1

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    volatility = (
        returns.std() * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Total Return": total_return,
    }


def run_strategy(df, top_n, lookback):

    data = df.copy()

    factor = f"return_{lookback}d"

    # Make sure the requested momentum factor exists.
    if factor not in data.columns:

        data[factor] = (
            data.groupby("ticker")["Adj Close"]
            .transform(
                lambda x: x / x.shift(lookback) - 1
            )
        )

    # Remove rows where momentum cannot be calculated.
    data = data.dropna(subset=[factor])

    # Rank stocks cross-sectionally on each date.
    data["rank"] = (
        data.groupby("Date")[factor]
        .rank(method="first", pct=True)
    )

    # Select top N stocks.
    selected = (
        data.sort_values(
            ["Date", "rank"],
            ascending=[True, False]
        )
        .groupby("Date")
        .head(top_n)
        .copy()
    )

    # Equal-weight portfolio.
    selected["weight"] = (
        selected.groupby("Date")["ticker"]
        .transform(
            lambda x: 1 / len(x)
        )
    )

    # CRITICAL:
    # next_return was calculated BEFORE filtering.
    #
    # Therefore it represents the actual next trading-day
    # return after the formation date.
    selected["contribution"] = (
        selected["weight"]
        * selected["next_return"]
    )

    portfolio = (
        selected
        .groupby("Date")["contribution"]
        .sum()
        .rename("portfolio_return")
        .reset_index()
    )

    return portfolio


def run_robustness():

    print("Loading data...")

    df = load_data()

    results = []

    configurations = [

        ("Top 20 - 3M Momentum", 20, 63),
        ("Top 50 - 3M Momentum", 50, 63),
        ("Top 100 - 3M Momentum", 100, 63),

        ("Top 20 - 6M Momentum", 20, 126),
        ("Top 50 - 6M Momentum", 50, 126),
        ("Top 100 - 6M Momentum", 100, 126),

        ("Top 20 - 12M Momentum", 20, 252),
        ("Top 50 - 12M Momentum", 50, 252),
        ("Top 100 - 12M Momentum", 100, 252),
    ]

    for name, top_n, lookback in configurations:

        print(f"Testing: {name}")

        portfolio = run_strategy(
            df,
            top_n,
            lookback
        )

        metrics = calculate_metrics(
            portfolio["portfolio_return"]
        )

        metrics["Strategy"] = name
        metrics["Top N"] = top_n
        metrics["Lookback"] = lookback
        metrics["Observations"] = len(portfolio)

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Strategy",
            "Top N",
            "Lookback",
            "Observations",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
            "Total Return",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Robustness Analysis")
    print("===================")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_robustness()