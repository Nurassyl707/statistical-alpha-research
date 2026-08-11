from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS_PATH = Path(
    "data/processed/cross_sectional_ranked.csv"
)

PRICES_DIR = Path(
    "data/raw/prices"
)

OUTPUT_PATH = Path(
    "data/processed/factor_attribution.csv"
)

QUINTILES = 5


def load_signals():
    print("Loading signal data...")

    df = pd.read_csv(SIGNALS_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    required = [
        "Date",
        "ticker",
        "return_21d_rank",
        "return_63d_rank",
        "return_126d_rank",
        "return_252d_rank",
        "volatility_21d_rank",
        "volatility_63d_rank",
        "volatility_252d_rank",
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


def load_prices():
    print("Loading prices...")

    frames = []

    files = sorted(
        PRICES_DIR.glob("*.csv")
    )

    for file_path in files:

        if file_path.name == "download_summary.csv":
            continue

        ticker = file_path.stem

        data = pd.read_csv(file_path)

        if (
            "Date" not in data.columns
            or "Adj Close" not in data.columns
        ):
            continue

        data = data[
            [
                "Date",
                "Adj Close",
            ]
        ].copy()

        data["Date"] = pd.to_datetime(
            data["Date"]
        )

        data["ticker"] = ticker

        data = data.sort_values(
            "Date"
        )

        data["next_return"] = (
            data["Adj Close"].shift(-1)
            / data["Adj Close"]
            - 1
        )

        frames.append(data)

    if not frames:
        raise ValueError(
            "No valid price files found."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"Loaded {prices['ticker'].nunique()} stocks."
    )

    print(
        f"Price observations: {len(prices):,}"
    )

    print(
        "Price date range: "
        f"{prices['Date'].min().date()} to "
        f"{prices['Date'].max().date()}"
    )

    return prices


def construct_factor_signals(df):
    data = df.copy()

    # Same composite definitions as the
    # final factor_backtest.py.
    data["momentum_composite"] = (
        0.10 * data["return_21d_rank"]
        + 0.20 * data["return_63d_rank"]
        + 0.30 * data["return_126d_rank"]
        + 0.40 * data["return_252d_rank"]
    )

    data["low_volatility_composite"] = (
        data[
            [
                "volatility_21d_rank",
                "volatility_63d_rank",
                "volatility_252d_rank",
            ]
        ].mean(axis=1)
    )

    return data


def get_month_end_dates(data):
    monthly = (
        data.assign(
            month=data["Date"].dt.to_period("M")
        )
        .groupby("month")["Date"]
        .max()
    )

    return pd.DatetimeIndex(
        monthly.values
    )


def prepare_evaluation_data(
    signals,
    prices,
):
    data = signals.copy()

    # Use only monthly rebalance dates.
    rebalance_dates = get_month_end_dates(
        data
    )

    data = data[
        data["Date"].isin(
            rebalance_dates
        )
    ].copy()

    # Merge next trading-day returns.
    data = data.merge(
        prices[
            [
                "Date",
                "ticker",
                "next_return",
            ]
        ],
        on=[
            "Date",
            "ticker",
        ],
        how="left",
    )

    data = data.dropna(
        subset=["next_return"]
    ).copy()

    return data


def calculate_quintile_returns(
    data,
    signal_column,
    low_is_good=False,
):
    working = data[
        [
            "Date",
            "ticker",
            signal_column,
            "next_return",
        ]
    ].dropna().copy()

    # Cross-sectional quintiles are calculated
    # independently on every rebalance date.
    #
    # For momentum:
    #   Q1 = lowest momentum
    #   Q5 = highest momentum
    #
    # For low volatility:
    #   Q1 = lowest volatility
    #   Q5 = highest volatility

    working["quintile"] = (
        working.groupby("Date")[signal_column]
        .rank(
            method="first",
            pct=True,
        )
        .mul(QUINTILES)
        .apply(np.ceil)
        .clip(
            lower=1,
            upper=QUINTILES,
        )
        .astype(int)
    )

    quintile_returns = (
        working.groupby(
            [
                "Date",
                "quintile",
            ]
        )["next_return"]
        .mean()
        .unstack("quintile")
        .sort_index()
    )

    quintile_returns.columns = [
        f"Q{int(column)}"
        for column in quintile_returns.columns
    ]

    # Ensure all quintile columns exist.
    for i in range(1, QUINTILES + 1):

        column = f"Q{i}"

        if column not in quintile_returns.columns:
            quintile_returns[column] = np.nan

    quintile_returns = quintile_returns[
        [
            f"Q{i}"
            for i in range(1, QUINTILES + 1)
        ]
    ]

    # Factor spread.
    #
    # Momentum:
    #   highest minus lowest = Q5 - Q1
    #
    # Low volatility:
    #   lowest volatility minus highest =
    #   Q1 - Q5
    if low_is_good:

        quintile_returns["Spread"] = (
            quintile_returns["Q1"]
            - quintile_returns["Q5"]
        )

    else:

        quintile_returns["Spread"] = (
            quintile_returns["Q5"]
            - quintile_returns["Q1"]
        )

    return working, quintile_returns


def calculate_ic(
    data,
    signal_column,
):
    working = data[
        [
            "Date",
            signal_column,
            "next_return",
        ]
    ].dropna().copy()

    ic = (
        working.groupby("Date")
        .apply(
            lambda group:
            group[signal_column].corr(
                group["next_return"],
                method="spearman",
            ),
            include_groups=False,
        )
        .dropna()
    )

    return ic


def annualized_return(
    daily_or_period_returns
):
    returns = pd.Series(
        daily_or_period_returns
    ).dropna()

    if len(returns) == 0:
        return np.nan

    cumulative = (
        1 + returns
    ).prod()

    years = len(returns) / 12

    if years <= 0:
        return np.nan

    return (
        cumulative ** (1 / years)
        - 1
    )


def summarize_factor(
    strategy,
    data,
    signal_column,
    low_is_good=False,
):
    _, quintile_returns = (
        calculate_quintile_returns(
            data,
            signal_column,
            low_is_good=low_is_good,
        )
    )

    ic = calculate_ic(
        data,
        signal_column,
    )

    result = {
        "Strategy": strategy,
    }

    # Quintile statistics.
    for i in range(1, QUINTILES + 1):

        column = f"Q{i}"

        series = (
            quintile_returns[column]
            .dropna()
        )

        result[
            f"{column} Mean Return"
        ] = series.mean()

        result[
            f"{column} Annualized Return"
        ] = annualized_return(series)

    spread = (
        quintile_returns["Spread"]
        .dropna()
    )

    result["Spread Mean Return"] = (
        spread.mean()
    )

    result["Spread Annualized Return"] = (
        annualized_return(spread)
    )

    # Information coefficient.
    result["Mean IC"] = ic.mean()

    result["IC Std"] = ic.std()

    result["IC IR"] = (
        ic.mean() / ic.std()
        if ic.std() > 0
        else np.nan
    )

    result["Positive IC %"] = (
        (ic > 0).mean()
    )

    result["IC Observations"] = (
        len(ic)
    )

    result["Rebalance Dates"] = (
        quintile_returns.index.nunique()
    )

    return result, quintile_returns, ic


def main():

    signals = load_signals()

    prices = load_prices()

    signals = construct_factor_signals(
        signals
    )

    data = prepare_evaluation_data(
        signals,
        prices,
    )

    if data.empty:
        raise ValueError(
            "No observations available "
            "after evaluation-period filtering."
        )

    print()
    print(
        "Factor Attribution Evaluation"
    )
    print(
        "============================="
    )

    print(
        "Start:",
        data["Date"].min().date()
    )

    print(
        "End:  ",
        data["Date"].max().date()
    )

    print(
        "Rebalance dates:",
        data["Date"].nunique()
    )

    strategies = {
        "21D Momentum": (
            "return_21d_rank",
            False,
        ),

        "63D Momentum": (
            "return_63d_rank",
            False,
        ),

        "126D Momentum": (
            "return_126d_rank",
            False,
        ),

        "252D Momentum": (
            "return_252d_rank",
            False,
        ),

        "Momentum Composite": (
            "momentum_composite",
            False,
        ),

        "Low Volatility": (
            "low_volatility_composite",
            True,
        ),
    }

    results = []

    quintile_outputs = {}

    ic_outputs = {}

    for name, (
        signal_column,
        low_is_good,
    ) in strategies.items():

        print()
        print(
            f"Running: {name}"
        )

        required_data = data.dropna(
            subset=[
                signal_column,
                "next_return",
            ]
        ).copy()

        result, quintiles, ic = (
            summarize_factor(
                name,
                required_data,
                signal_column,
                low_is_good=low_is_good,
            )
        )

        results.append(result)

        quintile_outputs[name] = quintiles

        ic_outputs[name] = ic

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Strategy",

            "Q1 Mean Return",
            "Q2 Mean Return",
            "Q3 Mean Return",
            "Q4 Mean Return",
            "Q5 Mean Return",

            "Q1 Annualized Return",
            "Q2 Annualized Return",
            "Q3 Annualized Return",
            "Q4 Annualized Return",
            "Q5 Annualized Return",

            "Spread Mean Return",
            "Spread Annualized Return",

            "Mean IC",
            "IC Std",
            "IC IR",
            "Positive IC %",
            "IC Observations",

            "Rebalance Dates",
        ]
    ]

    print()
    print(
        "Factor Attribution Analysis"
    )
    print(
        "==========================="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Save detailed quintile returns.
    quintile_output = Path(
        "data/processed/"
        "factor_quintile_returns.csv"
    )

    quintile_frames = []

    for strategy, frame in (
        quintile_outputs.items()
    ):

        temp = frame.copy()

        temp["Strategy"] = strategy

        temp = temp.reset_index()

        quintile_frames.append(
            temp
        )

    quintile_df = pd.concat(
        quintile_frames,
        ignore_index=True,
    )

    quintile_df.to_csv(
        quintile_output,
        index=False,
    )

    # Save daily/monthly IC observations.
    ic_output = Path(
        "data/processed/"
        "factor_ic.csv"
    )

    ic_frames = []

    for strategy, series in (
        ic_outputs.items()
    ):

        temp = series.rename(
            "IC"
        ).reset_index()

        temp["Strategy"] = strategy

        ic_frames.append(
            temp
        )

    ic_df = pd.concat(
        ic_frames,
        ignore_index=True,
    )

    ic_df.to_csv(
        ic_output,
        index=False,
    )

    print()
    print(
        f"Saved summary to: {OUTPUT_PATH}"
    )

    print(
        f"Saved quintile returns to: "
        f"{quintile_output}"
    )

    print(
        f"Saved IC data to: "
        f"{ic_output}"
    )


if __name__ == "__main__":
    main()
