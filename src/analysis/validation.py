import pandas as pd
from pathlib import Path


SIGNALS_PATH = Path("data/processed/alpha_signals.csv")


def main():

    df = pd.read_csv(SIGNALS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["ticker", "Date"])

    print("Data validation")
    print("=" * 60)

    # 1. Duplicate observations
    duplicates = df.duplicated(["Date", "ticker"]).sum()

    print(f"Duplicate Date/Ticker rows: {duplicates}")

    # 2. Date range
    print(
        f"Date range: "
        f"{df['Date'].min().date()} -> "
        f"{df['Date'].max().date()}"
    )

    # 3. Number of securities
    counts = df.groupby("Date")["ticker"].nunique()

    print("\nStocks per date:")
    print(counts.describe())

    # 4. Price validity
    bad_prices = (df["Adj Close"] <= 0).sum()

    print(f"\nNon-positive prices: {bad_prices}")

    # 5. Extreme returns
    if "return_252d" in df.columns:

        extreme = df["return_252d"].abs() > 5

        print(
            f"252D returns with abs(return) > 500%: "
            f"{extreme.sum()}"
        )

        if extreme.any():

            print("\nExtreme observations:")

            print(
                df.loc[
                    extreme,
                    [
                        "Date",
                        "ticker",
                        "Adj Close",
                        "return_252d"
                    ]
                ]
                .sort_values(
                    "return_252d",
                    ascending=False
                )
                .head(20)
                .to_string(index=False)
            )

    # 6. Missing values
    print("\nMissing values:")

    print(
        df[
            ["Date", "ticker", "Adj Close"]
        ].isna().sum()
    )

    print("\nValidation complete.")


if __name__ == "__main__":
    main()