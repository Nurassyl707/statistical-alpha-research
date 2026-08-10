from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS_PATH = Path(
    "data/processed/cross_sectional_ranked.csv"
)

PRICES_DIR = Path(
    "data/raw/prices"
)


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


def load_data():
    print("Loading signal data...")

    df = pd.read_csv(SIGNALS_PATH)

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


def build_signal(
    df,
    factors,
):
    data = df.copy()

    data["signal"] = (
        data[factors]
        .mean(axis=1)
    )

    return data


def run_portfolio(
    df,
    signal_column="signal",
    top_quantile=0.10,
):
    data = df.dropna(
        subset=[signal_column]
    ).copy()

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
        1
        / selected.groupby("Date")
        ["ticker"]
        .transform("count")
    )

    # Load prices needed to calculate
    # next-day returns.
    frames = []

    for ticker in selected["ticker"].unique():

        path = (
            PRICES_DIR
            / f"{ticker}.csv"
        )

        if not path.exists():
            continue

        price = pd.read_csv(path)

        if (
            "Date" not in price.columns
            or "Adj Close" not in price.columns
        ):
            continue

        price = price[
            ["Date", "Adj Close"]
        ].copy()

        price["Date"] = pd.to_datetime(
            price["Date"]
        )

        price["ticker"] = ticker

        price["next_return"] = (
            price.groupby("ticker")
            ["Adj Close"]
            .shift(-1)
            / price["Adj Close"]
            - 1
        )

        frames.append(price)

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    selected = selected.merge(
        prices[
            [
                "Date",
                "ticker",
                "next_return",
            ]
        ],
        on=["Date", "ticker"],
        how="left",
    )

    selected["contribution"] = (
        selected["weight"]
        * selected["next_return"]
    )

    portfolio = (
        selected.groupby("Date")
        ["contribution"]
        .sum()
    )

    return portfolio


def main():

    df = load_data()

    strategies = {
        "21D Momentum": [
            "return_21d_rank",
        ],

        "63D Momentum": [
            "return_63d_rank",
        ],

        "126D Momentum": [
            "return_126d_rank",
        ],

        "252D Momentum": [
            "return_252d_rank",
        ],

        "Momentum Composite": [
            "return_21d_rank",
            "return_63d_rank",
            "return_126d_rank",
            "return_252d_rank",
        ],

        "Low Volatility": [
            "volatility_21d_rank",
            "volatility_63d_rank",
            "volatility_252d_rank",
        ],
    }

    results = []

    for name, factors in strategies.items():

        print(
            f"Running: {name}"
        )

        data = build_signal(
            df,
            factors,
        )

        returns = run_portfolio(
            data
        )

        metrics = calculate_metrics(
            returns
        )

        metrics["Strategy"] = name

        results.append(metrics)

    results_df = (
        pd.DataFrame(results)
        [
            [
                "Strategy",
                "CAGR",
                "Volatility",
                "Sharpe",
                "Max Drawdown",
                "Total Return",
            ]
        ]
    )

    print()
    print(
        "Factor Attribution Analysis"
    )
    print(
        "==========================="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    output = Path(
        "data/processed/"
        "factor_attribution.csv"
    )

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