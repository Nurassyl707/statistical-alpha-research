from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("data/processed/backtest_returns.csv")
OUTPUT_DIR = Path("reports/figures")

MONTHLY_CSV = Path("data/processed/monthly_returns.csv")
MONTHLY_PLOT = OUTPUT_DIR / "monthly_returns.png"
ANNUAL_PLOT = OUTPUT_DIR / "annual_returns.png"


def load_returns():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"Date", "portfolio_return"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def calculate_monthly_returns(df):
    df = df.copy()

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    monthly = (
        df.groupby(["year", "month"])["portfolio_return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index()
    )

    monthly["Date"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str)
        + "-01"
    )

    monthly = monthly.sort_values("Date")

    return monthly


def calculate_annual_returns(df):
    annual = (
        df.groupby(df["Date"].dt.year)["portfolio_return"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="annual_return")
    )

    annual = annual.rename(
        columns={"Date": "year"}
    )

    return annual


def create_monthly_returns():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_returns()

    monthly = calculate_monthly_returns(df)

    annual = calculate_annual_returns(df)

    monthly.to_csv(
        MONTHLY_CSV,
        index=False,
    )

    print()
    print("Monthly Return Analysis")
    print("=======================")

    print(
        f"Months: {len(monthly):,}"
    )

    print(
        f"Best month: "
        f"{monthly['portfolio_return'].max():.2%}"
    )

    print(
        f"Worst month: "
        f"{monthly['portfolio_return'].min():.2%}"
    )

    print(
        f"Positive months: "
        f"{(monthly['portfolio_return'] > 0).sum():,}"
    )

    print(
        f"Negative months: "
        f"{(monthly['portfolio_return'] < 0).sum():,}"
    )

    print(
        f"Monthly win rate: "
        f"{(monthly['portfolio_return'] > 0).mean():.2%}"
    )

    print()
    print("Annual returns:")
    print(
        annual.to_string(index=False)
    )

    # Monthly return chart.
    plt.figure(figsize=(12, 6))

    plt.bar(
        monthly["Date"],
        monthly["portfolio_return"] * 100,
        width=20,
    )

    plt.axhline(
        y=0,
        linewidth=0.8,
    )

    plt.title("Monthly Strategy Returns")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        MONTHLY_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Annual return chart.
    plt.figure(figsize=(10, 6))

    plt.bar(
        annual["year"].astype(str),
        annual["annual_return"] * 100,
    )

    plt.axhline(
        y=0,
        linewidth=0.8,
    )

    plt.title("Annual Strategy Returns")
    plt.xlabel("Year")
    plt.ylabel("Return (%)")

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        ANNUAL_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Saved monthly data to: {MONTHLY_CSV}"
    )

    print(
        f"Saved monthly chart to: {MONTHLY_PLOT}"
    )

    print(
        f"Saved annual chart to: {ANNUAL_PLOT}"
    )

    return monthly, annual


def validate_monthly_returns():
    df = load_returns()

    monthly = calculate_monthly_returns(df)

    annual = calculate_annual_returns(df)

    print()
    print("Monthly Return Validation")
    print("=========================")

    print(
        f"Trading days: {len(df):,}"
    )

    print(
        f"Months: {len(monthly):,}"
    )

    print(
        f"Annual periods: {len(annual):,}"
    )

    print(
        f"Missing monthly returns: "
        f"{monthly['portfolio_return'].isna().sum():,}"
    )

    print()
    print("Best 10 months:")

    print(
        monthly
        .sort_values(
            "portfolio_return",
            ascending=False,
        )
        .head(10)
        [["Date", "portfolio_return"]]
        .to_string(index=False)
    )

    print()
    print("Worst 10 months:")

    print(
        monthly
        .sort_values(
            "portfolio_return",
            ascending=True,
        )
        .head(10)
        [["Date", "portfolio_return"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    create_monthly_returns()