from pathlib import Path

import pandas as pd


IC_PATH = Path(
    "data/processed/factor_ic.csv"
)

OUTPUT_PATH = Path(
    "data/processed/factor_correlation.csv"
)


def load_ic_data():
    print("Loading factor IC data...")

    df = pd.read_csv(IC_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    required = [
        "Date",
        "IC",
        "Strategy",
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


def build_correlation_matrix(df):
    ic_matrix = (
        df.pivot(
            index="Date",
            columns="Strategy",
            values="IC",
        )
        .sort_index()
    )

    correlation = (
        ic_matrix
        .corr(method="pearson")
    )

    return ic_matrix, correlation


def main():

    df = load_ic_data()

    ic_matrix, correlation = (
        build_correlation_matrix(df)
    )

    print()
    print(
        "Factor IC Correlation"
    )
    print(
        "===================="
    )

    print(
        correlation.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print(
        "IC Observations by Factor"
    )
    print(
        "========================="
    )

    print(
        ic_matrix.count()
        .to_string()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlation.to_csv(
        OUTPUT_PATH
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
