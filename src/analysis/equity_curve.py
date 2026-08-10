from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


BACKTEST_FILE = Path(
    "data/processed/backtest_returns.csv"
)

OUTPUT_DIR = Path(
    "reports/figures"
)

OUTPUT_FILE = OUTPUT_DIR / "strategy_vs_spy.png"


def load_strategy():

    df = pd.read_csv(
        BACKTEST_FILE
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    df["strategy_wealth"] = (
        1 + df["portfolio_return"]
    ).cumprod()

    return df[
        ["Date", "strategy_wealth"]
    ]


def load_spy():

    print("Downloading SPY data...")

    spy = yf.download(
        "SPY",
        start="2021-01-01",
        end="2026-08-08",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(
        spy.columns,
        pd.MultiIndex
    ):
        spy.columns = (
            spy.columns
            .get_level_values(0)
        )

    spy = spy[
        ["Adj Close"]
    ].copy()

    spy.index = pd.to_datetime(
        spy.index
    )

    spy["spy_return"] = (
        spy["Adj Close"]
        .pct_change()
    )

    spy = spy.dropna()

    spy["spy_wealth"] = (
        1 + spy["spy_return"]
    ).cumprod()

    spy = spy.reset_index()

    return spy[
        ["Date", "spy_wealth"]
    ]


def create_equity_curve():

    strategy = load_strategy()

    spy = load_spy()

    df = pd.merge(
        strategy,
        spy,
        on="Date",
        how="inner"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        df["Date"],
        df["strategy_wealth"],
        label="Strategy"
    )

    plt.plot(
        df["Date"],
        df["spy_wealth"],
        label="SPY"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Growth of $1"
    )

    plt.title(
        "Momentum + Low Volatility Strategy vs SPY"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200
    )

    plt.close()

    print()
    print(
        "Equity curve created."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_equity_curve()