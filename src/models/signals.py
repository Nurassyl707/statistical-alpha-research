from pathlib import Path

import pandas as pd


MOMENTUM_WEIGHTS = {
    "return_21d_rank": 0.10,
    "return_63d_rank": 0.20,
    "return_126d_rank": 0.30,
    "return_252d_rank": 0.40,
}

VOLATILITY_COLUMNS = [
    "volatility_21d_rank",
    "volatility_63d_rank",
    "volatility_252d_rank",
]

MOMENTUM_WEIGHT = 0.75
VOLATILITY_WEIGHT = 0.25


def build_alpha_signal(
    input_file: str = "data/processed/cross_sectional_ranked.csv",
    output_file: str = "data/processed/alpha_signals.csv",
) -> pd.DataFrame:
    """
    Build a composite cross-sectional alpha signal.

    The signal combines:
        1. Multi-horizon momentum
        2. Low-volatility exposure

    All input factors are already cross-sectional percentile ranks.
    Higher values represent more attractive characteristics.
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
        *MOMENTUM_WEIGHTS.keys(),
        *VOLATILITY_COLUMNS,
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
    # Momentum score
    # ---------------------------------------------------------

    momentum_available = data[
        list(MOMENTUM_WEIGHTS.keys())
    ].notna().all(axis=1)

    data["momentum_score"] = 0.0

    for column, weight in MOMENTUM_WEIGHTS.items():
        data["momentum_score"] += (
            data[column].fillna(0.0) * weight
        )

    data.loc[
        ~momentum_available,
        "momentum_score"
    ] = float("nan")

    # ---------------------------------------------------------
    # Low-volatility score
    # ---------------------------------------------------------

    volatility_available = data[
        VOLATILITY_COLUMNS
    ].notna().all(axis=1)

    data["low_volatility_score"] = (
        data[VOLATILITY_COLUMNS]
        .mean(axis=1)
    )

    data.loc[
        ~volatility_available,
        "low_volatility_score"
    ] = float("nan")

    # ---------------------------------------------------------
    # Composite alpha score
    # ---------------------------------------------------------

    signal_available = (
        momentum_available
        & volatility_available
    )

    data["alpha_score"] = (
        MOMENTUM_WEIGHT * data["momentum_score"]
        + VOLATILITY_WEIGHT * data["low_volatility_score"]
    )

    data.loc[
        ~signal_available,
        "alpha_score"
    ] = float("nan")

    # ---------------------------------------------------------
    # Rank the final alpha score cross-sectionally
    # ---------------------------------------------------------

    data["alpha_rank"] = (
        data.groupby("Date")["alpha_score"]
        .rank(method="average", pct=True)
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
    print("Alpha signal construction complete.")
    print(
        f"Rows: {len(data):,}"
    )
    print(
        f"Stocks: {data['ticker'].nunique():,}"
    )
    print(
        f"Dates: "
        f"{data['Date'].min().date()} "
        f"to "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Signal weights:")
    print(
        f"Momentum: {MOMENTUM_WEIGHT:.0%}"
    )
    print(
        f"Low volatility: {VOLATILITY_WEIGHT:.0%}"
    )

    print()
    print(
        f"Valid alpha observations: "
        f"{data['alpha_score'].notna().sum():,}"
    )

    print(
        f"Missing alpha observations: "
        f"{data['alpha_score'].isna().sum():,}"
    )

    print()
    print(f"Saved to: {output_path}")

    return data


def validate_alpha_signal(
    input_file: str = "data/processed/alpha_signals.csv",
) -> None:
    """
    Validate the generated alpha signal.
    """

    data = pd.read_csv(input_file)

    data["Date"] = pd.to_datetime(data["Date"])

    print()
    print("Alpha signal validation")
    print("=======================")

    for column in [
        "momentum_score",
        "low_volatility_score",
        "alpha_score",
        "alpha_rank",
    ]:
        series = data[column]

        print(
            f"{column}: "
            f"min={series.min():.4f}, "
            f"max={series.max():.4f}, "
            f"mean={series.mean():.4f}, "
            f"missing={series.isna().sum():,}"
        )

    print()
    print("Top 10 alpha scores on latest date:")

    latest_date = data["Date"].max()

    latest = data[
        data["Date"] == latest_date
    ].sort_values(
        "alpha_score",
        ascending=False,
    )

    print(
        latest[
            [
                "ticker",
                "momentum_score",
                "low_volatility_score",
                "alpha_score",
                "alpha_rank",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    build_alpha_signal()
    validate_alpha_signal()