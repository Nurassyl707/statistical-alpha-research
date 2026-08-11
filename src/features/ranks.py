from pathlib import Path

import pandas as pd


RETURN_FACTORS = [
    "return_21d",
    "return_63d",
    "return_126d",
    "return_252d",
]

VOLATILITY_FACTORS = [
    "volatility_21d",
    "volatility_63d",
    "volatility_252d",
]


def add_cross_sectional_ranks(
    input_file: str = "data/processed/cross_sectional.csv",
    output_file: str = "data/processed/cross_sectional_ranked.csv",
) -> pd.DataFrame:
    """
    Add cross-sectional percentile ranks for each factor.

    Higher return = higher rank.
    Lower volatility = higher rank.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    required_columns = [
        "Date",
        "ticker",
        *RETURN_FACTORS,
        *VOLATILITY_FACTORS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["Date"] = pd.to_datetime(data["Date"])

    # ---------------------------------------------------------
    # Return factors
    # Higher return = higher rank
    # ---------------------------------------------------------

    for column in RETURN_FACTORS:

        rank_column = f"{column}_rank"

        data[rank_column] = (
            data.groupby("Date")[column]
            .rank(
                method="average",
                pct=True,
                ascending=True,
            )
        )

    # ---------------------------------------------------------
    # Volatility factors
    # Lower volatility = higher rank
    # ---------------------------------------------------------

    for column in VOLATILITY_FACTORS:

        rank_column = f"{column}_rank"

        normal_rank = (
            data.groupby("Date")[column]
            .rank(
                method="average",
                pct=True,
                ascending=True,
            )
        )

        data[rank_column] = (
            1.0 - normal_rank
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Cross-sectional ranking complete.")
    print("===============================")

    print(f"Rows: {len(data):,}")
    print(f"Stocks: {data['ticker'].nunique():,}")

    print(
        f"Dates: "
        f"{data['Date'].min().date()} "
        f"to "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Rank columns:")

    rank_columns = [
        f"{column}_rank"
        for column in (
            RETURN_FACTORS
            + VOLATILITY_FACTORS
        )
    ]

    for column in rank_columns:
        print(f"  {column}")

    print()
    print(f"Saved to: {output_path}")

    return data


def validate_ranks(
    input_file: str = "data/processed/cross_sectional_ranked.csv",
) -> None:
    """
    Validate generated cross-sectional ranks.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ranked dataset not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    rank_columns = [
        f"{column}_rank"
        for column in (
            RETURN_FACTORS
            + VOLATILITY_FACTORS
        )
    ]

    print()
    print("Rank validation")
    print("================")

    validation_failed = False

    for column in rank_columns:

        if column not in data.columns:
            print(f"{column}: MISSING")
            validation_failed = True
            continue

        series = data[column]
        non_null = series.dropna()

        if non_null.empty:
            print(f"{column}: NO VALID VALUES")
            validation_failed = True
            continue

        minimum = non_null.min()
        maximum = non_null.max()
        mean = non_null.mean()
        missing = series.isna().sum()

        valid_range = (
            minimum >= 0.0
            and maximum <= 1.0
        )

        if not valid_range:
            validation_failed = True

        status = "OK" if valid_range else "INVALID"

        print(
            f"{column}: "
            f"min={minimum:.4f}, "
            f"max={maximum:.4f}, "
            f"mean={mean:.4f}, "
            f"missing={missing:,} "
            f"[{status}]"
        )

    print()

    if validation_failed:
        raise ValueError(
            "Rank validation failed."
        )

    print("All rank columns passed validation.")


if __name__ == "__main__":

    add_cross_sectional_ranks()

    validate_ranks