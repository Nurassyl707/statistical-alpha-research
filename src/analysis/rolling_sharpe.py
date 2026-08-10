from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("data/processed/backtest_returns.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_FILE = OUTPUT_DIR / "rolling_sharpe.png"

TRADING_DAYS = 252
ROLLING_WINDOW = 252


def create_rolling_sharpe():
    """
    Calculate and plot the 252-day rolling annualized Sharpe ratio.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "Date",
        "portfolio_return",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    returns = df["portfolio_return"]

    # Rolling mean and standard deviation.
    rolling_mean = (
        returns
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    rolling_std = (
        returns
        .rolling(ROLLING_WINDOW)
        .std()
    )

    # Annualized Sharpe ratio.
    df["rolling_sharpe"] = (
        rolling_mean / rolling_std
    ) * (TRADING_DAYS ** 0.5)

    valid = df["rolling_sharpe"].dropna()

    print()
    print("Rolling Sharpe Analysis")
    print("=======================")

    print(f"Window: {ROLLING_WINDOW} trading days")
    print(f"Valid observations: {len(valid):,}")

    print(
        f"Minimum rolling Sharpe: "
        f"{valid.min():.4f}"
    )

    print(
        f"Maximum rolling Sharpe: "
        f"{valid.max():.4f}"
    )

    print(
        f"Mean rolling Sharpe: "
        f"{valid.mean():.4f}"
    )

    print(
        f"Median rolling Sharpe: "
        f"{valid.median():.4f}"
    )

    print(
        f"Latest rolling Sharpe: "
        f"{valid.iloc[-1]:.4f}"
    )

    # Plot.
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["rolling_sharpe"],
        linewidth=1.5,
    )

    plt.axhline(
        y=0,
        linewidth=0.8,
    )

    plt.axhline(
        y=1,
        linewidth=0.8,
        linestyle="--",
    )

    plt.title(
        "252-Day Rolling Sharpe Ratio"
    )

    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")

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

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    return df


def validate_rolling_sharpe():
    """
    Validate the rolling Sharpe calculation.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    rolling_mean = (
        df["portfolio_return"]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    rolling_std = (
        df["portfolio_return"]
        .rolling(ROLLING_WINDOW)
        .std()
    )

    df["rolling_sharpe"] = (
        rolling_mean / rolling_std
    ) * (TRADING_DAYS ** 0.5)

    valid = df["rolling_sharpe"].dropna()

    print()
    print("Rolling Sharpe Validation")
    print("=========================")

    print(f"Rows: {len(df):,}")

    print(
        f"Date range: "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Valid rolling observations: "
        f"{len(valid):,}"
    )

    print(
        f"Missing rolling Sharpe: "
        f"{df['rolling_sharpe'].isna().sum():,}"
    )

    print(
        f"Minimum: {valid.min():.4f}"
    )

    print(
        f"Maximum: {valid.max():.4f}"
    )

    print(
        f"Mean: {valid.mean():.4f}"
    )

    print(
        f"Median: {valid.median():.4f}"
    )

    print()
    print("Last 10 observations:")

    print(
        df[
            [
                "Date",
                "portfolio_return",
                "rolling_sharpe",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    create_rolling_sharpe()