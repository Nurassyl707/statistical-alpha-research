from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("data/processed/backtest_returns.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_FILE = OUTPUT_DIR / "drawdown.png"


def create_drawdown_chart():
    """
    Create a drawdown chart from the strategy backtest.

    Drawdown is measured relative to the running historical
    maximum of cumulative wealth.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "Date",
        "cumulative_return",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # Running peak of portfolio wealth.
    running_peak = df["cumulative_return"].cummax()

    # Drawdown relative to the historical peak.
    df["drawdown"] = (
        df["cumulative_return"] / running_peak
    ) - 1.0

    # Maximum drawdown.
    max_drawdown = df["drawdown"].min()

    max_dd_idx = df["drawdown"].idxmin()

    max_dd_date = df.loc[max_dd_idx, "Date"]

    print()
    print("Drawdown Analysis")
    print("=================")
    print(f"Maximum drawdown: {max_drawdown:.2%}")
    print(f"Maximum drawdown date: {max_dd_date.date()}")

    # Plot.
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["drawdown"] * 100,
        linewidth=1.5,
    )

    plt.axhline(
        y=0,
        linewidth=0.8,
    )

    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved to: {OUTPUT_FILE}")

    return df


def validate_drawdown():
    """
    Validate the drawdown calculation.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    running_peak = df["cumulative_return"].cummax()

    df["drawdown"] = (
        df["cumulative_return"] / running_peak
    ) - 1.0

    print()
    print("Drawdown Validation")
    print("===================")

    print(f"Rows: {len(df):,}")
    print(
        f"Date range: "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Maximum drawdown: "
        f"{df['drawdown'].min():.2%}"
    )

    print(
        f"Current drawdown: "
        f"{df['drawdown'].iloc[-1]:.2%}"
    )

    print(
        f"Number of observations below zero: "
        f"{(df['drawdown'] < 0).sum():,}"
    )

    print()
    print("Worst 10 drawdown observations:")

    print(
        df[
            ["Date", "cumulative_return", "drawdown"]
        ]
        .sort_values("drawdown")
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    create_drawdown_chart()