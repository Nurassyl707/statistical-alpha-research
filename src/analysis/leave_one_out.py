import pandas as pd
import numpy as np
from pathlib import Path


SIGNALS_PATH = Path("data/processed/alpha_signals.csv")


def calculate_metrics(returns):

    returns = pd.Series(returns).dropna()

    equity = (1 + returns).cumprod()

    years = len(returns) / 252

    total_return = equity.iloc[-1] - 1

    cagr = equity.iloc[-1] ** (1 / years) - 1

    volatility = returns.std() * np.sqrt(252)

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
        "Total Return": total_return,
    }


def run_strategy(df, top_n, lookback):

    data = df.copy()

    factor = f"return_{lookback}d"

    data = data.dropna(subset=[factor])

    data["rank"] = (
        data.groupby("Date")[factor]
        .rank(method="first", pct=True)
    )

    selected = (
        data.sort_values(
            ["Date", "rank"],
            ascending=[True, False]
        )
        .groupby("Date")
        .head(top_n)
        .copy()
    )

    selected["weight"] = (
        selected.groupby("Date")["ticker"]
        .transform(lambda x: 1 / len(x))
    )

    selected["next_return"] = (
        selected.groupby("ticker")["Adj Close"].shift(-1)
        / selected["Adj Close"]
        - 1
    )

    selected["contribution"] = (
        selected["weight"] *
        selected["next_return"]
    )

    portfolio = (
        selected.groupby("Date")["contribution"]
        .sum()
    )

    return portfolio


def main():

    print("Loading data...")

    df = pd.read_csv(SIGNALS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    configurations = [
        ("Top 20 - 3M", 20, 63),
        ("Top 50 - 3M", 50, 63),
        ("Top 20 - 6M", 20, 126),
        ("Top 50 - 6M", 50, 126),
        ("Top 20 - 12M", 20, 252),
        ("Top 50 - 12M", 50, 252),
    ]

    results = []

    for name, top_n, lookback in configurations:

        for exclusion in ["None", "SNDK"]:

            test_df = df.copy()

            if exclusion == "SNDK":
                test_df = test_df[
                    test_df["ticker"] != "SNDK"
                ]

            portfolio = run_strategy(
                test_df,
                top_n,
                lookback
            )

            metrics = calculate_metrics(portfolio)

            metrics["Strategy"] = name
            metrics["Exclusion"] = exclusion

            results.append(metrics)

    results_df = pd.DataFrame(results)

    print()
    print("Leave-One-Out Robustness")
    print("========================")

    print(
        results_df[
            [
                "Strategy",
                "Exclusion",
                "CAGR",
                "Volatility",
                "Sharpe",
                "Max Drawdown",
                "Total Return",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()