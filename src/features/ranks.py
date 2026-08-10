from pathlib import Path

import pandas as pd


# Factors that we want to rank cross-sectionally.
# Higher return = better.
# Lower volatility = better.
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

    For every trading date, stocks are ranked relative to all other
    stocks available on that same date.

    Return factors:
        Higher return -> higher rank.

    Volatility factors:
        Lower volatility -> higher rank.

    Missing factor values remain NaN.
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
    # Cross-sectional percentile ranks
    # ---------------------------------------------------------
    #
    # pct=True converts ranks into values between 0 and 1.
    #
    # Example:
    #
    # 1.00 -> among the strongest stocks
    # 0.50 -> around the middle
    # 0.01 -> among the weakest stocks
    #
    # Ranking is performed independently for every date.
    # ---------------------------------------------------------

    for column in RETURN_FACTORS:
        rank_column = f"{column}_rank"

        data[rank_column] = (
            data.groupby("Date")[column]
            .rank(method="average", pct=True)
        )

    # ---------------------------------------------------------
    # Volatility ranking
    # ---------------------------------------------------------
    #
    # For volatility, LOWER volatility is considered better.
    #
    # Therefore:
    #
    # low volatility -> high rank
    # high volatility -> low rank
    #
    # We calculate the normal percentile rank and then invert it.
    # ---------------------------------------------------------

    for column in VOLATILITY_FACTORS:
        rank_column = f"{column}_rank"

        data[rank_column] = (
            1
            - data.groupby("Date")[column]
            .rank(method="average", pct=True)
            + (
                data.groupby("Date")[column]
                .transform("count") > 0
            ) * 0
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(output_path, index=False)

    print()
    print("Cross-sectional ranking complete.")
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
        for column in RETURN_FACTORS + VOLATILITY_FACTORS
    ]

    print(rank_columns)

    print()
    print(f"Saved to: {output_path}")

    return data


def validate_ranks(
    input_file: str = "data/processed/cross_sectional_ranked.csv",
) -> None:
    """
    Validate the generated cross-sectional ranks.
    """

    data = pd.read_csv(input_file)

    rank_columns = [
        f"{column}_rank"
        for column in RETURN_FACTORS + VOLATILITY_FACTORS
    ]

    print()
    print("Rank validation")
    print("================")

    for column in rank_columns:
        if column not in data.columns:
            print(f"{column}: MISSING")
            continue

        series = data[column]

        non_null = series.dropna()

        if non_null.empty:
            print(f"{column}: NO VALID VALUES")
            continue

        print(
            f"{column}: "
            f"min={non_null.min():.4f}, "
            f"max={non_null.max():.4f}, "
            f"mean={non_null.mean():.4f}, "
            f"missing={series.isna().sum():,}"
        )


if __name__ == "__main__":
    add_cross_sectional_ranks()
    validate_ranks()