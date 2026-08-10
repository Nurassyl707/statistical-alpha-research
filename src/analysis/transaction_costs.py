from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_PATH = Path(
    "data/processed/backtest_returns.csv"
)

TURNOVER_PATH = Path(
    "data/processed/turnover.csv"
)

OUTPUT_PATH = Path(
    "data/processed/transaction_costs.csv"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

COST_BPS = [
    5,
    10,
    20,
    30,
    50,
]


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():
    backtest = pd.read_csv(
        BACKTEST_PATH
    )

    turnover = pd.read_csv(
        TURNOVER_PATH
    )

    backtest["Date"] = pd.to_datetime(
        backtest["Date"]
    )

    turnover["Date"] = pd.to_datetime(
        turnover["Date"]
    )

    return backtest, turnover


# ------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------

def calculate_metrics(returns):

    returns = pd.Series(
        returns
    ).dropna()

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


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def main():

    print("Loading data...")

    backtest, turnover = load_data()

    # --------------------------------------------------------
    # Merge turnover onto daily backtest returns.
    # --------------------------------------------------------

    data = backtest.merge(
        turnover[
            [
                "Date",
                "one_way_turnover",
            ]
        ],
        on="Date",
        how="left",
    )

    data["one_way_turnover"] = (
        data["one_way_turnover"]
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Calculate transaction-cost-adjusted returns.
    #
    # Cost = turnover × cost per unit traded.
    #
    # Example:
    # 47.8% turnover × 10 bps
    # = 0.0478 × 0.001
    # = 0.00478%
    # --------------------------------------------------------

    results = []

    for cost_bps in COST_BPS:

        cost_rate = (
            cost_bps / 10_000
        )

        data[
            "transaction_cost"
        ] = (
            data["one_way_turnover"]
            * cost_rate
        )

        data[
            "net_return"
        ] = (
            data["portfolio_return"]
            - data["transaction_cost"]
        )

        metrics = calculate_metrics(
            data["net_return"]
        )

        metrics["Cost (bps)"] = cost_bps

        metrics[
            "Average Annual Cost"
        ] = (
            data["transaction_cost"].mean()
            * 252
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Cost (bps)",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
            "Total Return",
            "Average Annual Cost",
        ]
    ]

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display.
    # --------------------------------------------------------

    print()
    print(
        "Transaction Cost Analysis"
    )
    print(
        "========================="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()