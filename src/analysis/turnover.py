from pathlib import Path

import numpy as np
import pandas as pd


PORTFOLIO_PATH = Path(
    "data/processed/portfolio_weights.csv"
)

OUTPUT_PATH = Path(
    "data/processed/turnover.csv"
)


def load_portfolio(
    input_path: str = str(PORTFOLIO_PATH),
) -> pd.DataFrame:
    """
    Load portfolio weights and validate structure.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio file not found: {path}"
        )

    portfolio = pd.read_csv(path)

    required_columns = [
        "Date",
        "ticker",
        "weight",
    ]

    missing = [
        column
        for column in required_columns
        if column not in portfolio.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    portfolio["Date"] = pd.to_datetime(
        portfolio["Date"]
    )

    portfolio["weight"] = pd.to_numeric(
        portfolio["weight"],
        errors="coerce",
    )

    if portfolio["weight"].isna().any():
        raise ValueError(
            "Portfolio contains invalid weights."
        )

    return portfolio[
        ["Date", "ticker", "weight"]
    ].copy()


def calculate_turnover(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate one-way and two-way portfolio turnover.

    Turnover is calculated as:

        one-way turnover =
            0.5 * sum(|new_weight - old_weight|)

    The first portfolio has no previous portfolio,
    so turnover is reported as NaN.
    """

    dates = sorted(
        portfolio["Date"].unique()
    )

    results = []

    previous_weights = None

    for date in dates:

        current = portfolio[
            portfolio["Date"] == date
        ][
            ["ticker", "weight"]
        ].copy()

        current_weights = (
            current
            .set_index("ticker")["weight"]
        )

        if previous_weights is None:

            one_way_turnover = np.nan
            two_way_turnover = np.nan

        else:

            combined = (
                pd.concat(
                    [
                        previous_weights,
                        current_weights,
                    ],
                    axis=1,
                )
                .fillna(0.0)
            )

            combined.columns = [
                "previous_weight",
                "current_weight",
            ]

            weight_change = (
                combined["current_weight"]
                - combined["previous_weight"]
            )

            two_way_turnover = (
                weight_change.abs().sum()
            )

            one_way_turnover = (
                0.5 * two_way_turnover
            )

        results.append(
            {
                "Date": date,
                "one_way_turnover":
                    one_way_turnover,
                "two_way_turnover":
                    two_way_turnover,
                "holdings":
                    len(current_weights),
            }
        )

        previous_weights = current_weights

    result = pd.DataFrame(results)

    return result


def main():

    print("Loading portfolio...")

    portfolio = load_portfolio()

    print(
        f"Portfolio observations: "
        f"{len(portfolio):,}"
    )

    print(
        f"Rebalance dates: "
        f"{portfolio['Date'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{portfolio['Date'].min().date()} "
        f"to "
        f"{portfolio['Date'].max().date()}"
    )

    turnover = calculate_turnover(
        portfolio
    )

    print()
    print("Portfolio Turnover")
    print("==================")

    print(
        turnover.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    valid = turnover[
        "one_way_turnover"
    ].dropna()

    print()
    print("Turnover Summary")
    print("================")

    print(
        f"Average one-way turnover: "
        f"{valid.mean():.2%}"
    )

    print(
        f"Median one-way turnover: "
        f"{valid.median():.2%}"
    )

    print(
        f"Maximum one-way turnover: "
        f"{valid.max():.2%}"
    )

    print(
        f"Minimum one-way turnover: "
        f"{valid.min():.2%}"
    )

    print(
        f"Average two-way turnover: "
        f"{turnover['two_way_turnover'].dropna().mean():.2%}"
    )

    output = Path(
        OUTPUT_PATH
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    turnover.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Saved to: {output}"
    )


if __name__ == "__main__":
    main()