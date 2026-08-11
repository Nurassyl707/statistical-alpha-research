
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SIGNAL_FILE = Path("data/processed/alpha_signals.csv")
PRICE_DIR = Path("data/raw/prices")

OUTPUT_DIR = Path("data/processed")

SUMMARY_FILE = OUTPUT_DIR / "multi_factor_backtest.csv"
MONTHLY_FILE = OUTPUT_DIR / "multi_factor_monthly.csv"
HOLDINGS_FILE = OUTPUT_DIR / "multi_factor_holdings.csv"


# Factor weights
MOMENTUM_WEIGHT = 0.70
LOW_VOL_WEIGHT = 0.30

# Select top 10% of the universe
TOP_QUANTILE = 0.10

# Transaction cost
TRANSACTION_COST_BPS = 10.0


# ============================================================
# Utility
# ============================================================

def find_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"Could not find any of: {candidates}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


# ============================================================
# Load signals
# ============================================================

def load_signal_data():
    print("Loading signal data...")

    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(
            f"Signal file not found: {SIGNAL_FILE}"
        )

    signals = pd.read_csv(SIGNAL_FILE)

    print(f"Signal columns: {signals.columns.tolist()}")

    date_col = find_column(
        signals,
        ["Date", "date"]
    )

    ticker_col = find_column(
        signals,
        ["ticker", "Ticker", "symbol", "Symbol"]
    )

    # Momentum
    momentum_col = find_column(
        signals,
        [
            "momentum",
            "Momentum",
            "momentum_score",
            "Momentum Score",
            "momentum_composite",
            "Momentum Composite",
        ]
    )

    # Volatility
    low_vol_col = find_column(
        signals,
        [
            "low_volatility_score",
            "Low Volatility Score",
            "low_volatility",
            "Low Volatility",
            "low_vol",
            "Low Vol",
            "volatility_score",
            "Volatility Score",
            "volatility",
            "Volatility",
        ]
    )

    signals = signals.rename(
        columns={
            date_col: "Date",
            ticker_col: "ticker",
            momentum_col: "momentum",
            low_vol_col: "low_volatility",
        }
    )

    signals["Date"] = pd.to_datetime(
        signals["Date"],
        errors="coerce"
    )

    signals["ticker"] = signals["ticker"].astype(str)

    signals["momentum"] = pd.to_numeric(
        signals["momentum"],
        errors="coerce"
    )

    signals["low_volatility"] = pd.to_numeric(
        signals["low_volatility"],
        errors="coerce"
    )

    signals = signals.dropna(
        subset=[
            "Date",
            "ticker",
            "momentum",
            "low_volatility",
        ]
    )

    signals = signals.sort_values(
        ["Date", "ticker"]
    )

    print(
        f"Loaded {len(signals):,} signal observations."
    )

    print(
        f"Signal date range: "
        f"{signals['Date'].min().date()} "
        f"to "
        f"{signals['Date'].max().date()}"
    )

    print(
        f"Signal stocks: "
        f"{signals['ticker'].nunique()}"
    )

    return signals


# ============================================================
# Load prices
# ============================================================

def load_prices():
    """
    Load historical adjusted prices from data/raw/prices/*.csv.

    Each file represents one ticker. The loader standardizes the
    data into:

        Date
        ticker
        price

    Adjusted Close is preferred when available.
    """

    print("Loading prices...")

    if not PRICE_DIR.exists():
        raise FileNotFoundError(
            f"Price directory not found: {PRICE_DIR}"
        )

    files = sorted(PRICE_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV price files found in {PRICE_DIR}"
        )

    frames = []

    for file in files:
        try:
            df = pd.read_csv(file)

            if df.empty:
                continue

            date_col = find_column(
                df,
                ["Date", "date"],
                required=False,
            )

            price_col = find_column(
                df,
                [
                    "Adj Close",
                    "adj_close",
                    "Adjusted Close",
                    "Close",
                    "close",
                ],
                required=False,
            )

            if date_col is None or price_col is None:
                continue

            ticker = file.stem.upper()

            temp = df[[date_col, price_col]].copy()

            temp = temp.rename(
                columns={
                    date_col: "Date",
                    price_col: "price",
                }
            )

            temp["ticker"] = ticker

            temp["Date"] = pd.to_datetime(
                temp["Date"],
                errors="coerce",
            )

            temp["price"] = pd.to_numeric(
                temp["price"],
                errors="coerce",
            )

            temp = temp.dropna(
                subset=[
                    "Date",
                    "price",
                ]
            )

            temp = temp[temp["price"] > 0]

            if not temp.empty:
                frames.append(
                    temp[
                        [
                            "Date",
                            "ticker",
                            "price",
                        ]
                    ]
                )

        except Exception as exc:
            print(
                f"Warning: could not load {file.name}: {exc}"
            )

    if not frames:
        raise ValueError(
            f"No valid price files found in {PRICE_DIR}"
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices = prices.drop_duplicates(
        subset=["Date", "ticker"],
        keep="last",
    )

    prices = prices.sort_values(
        ["ticker", "Date"]
    ).reset_index(drop=True)

    print(
        f"Loaded {prices['ticker'].nunique()} stocks."
    )

    print(
        f"Price observations: {len(prices):,}"
    )

    print(
        "Price date range: "
        f"{prices['Date'].min().date()} "
        f"to "
        f"{prices['Date'].max().date()}"
    )

    return prices


def get_price_matrix(prices):
    """
    Convert long-form price data into a Date x ticker matrix.

    Rows:
        Trading dates

    Columns:
        Tickers

    Values:
        Adjusted/selected price
    """

    price_matrix = prices.pivot(
        index="Date",
        columns="ticker",
        values="price",
    )

    price_matrix = price_matrix.sort_index()

    return price_matrix


# ============================================================
# Portfolio construction
# ============================================================

def construct_portfolio(signal_slice):
    """
    Construct a long-only multi-factor portfolio.

    Momentum:
        Higher is better.

    Volatility:
        Lower is better.

    Composite:
        70% momentum rank
        30% low-volatility rank

    Select the top 10%.
    """

    df = signal_slice.copy()

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "weight"]
        )

    df["momentum_rank"] = (
        df["momentum"]
        .rank(pct=True, method="average")
    )

    # Lower volatility = better.
    df["low_vol_rank"] = (
        (-df["low_volatility"])
        .rank(pct=True, method="average")
    )

    df["composite_score"] = (
        MOMENTUM_WEIGHT * df["momentum_rank"]
        +
        LOW_VOL_WEIGHT * df["low_vol_rank"]
    )

    df = df.dropna(
        subset=["composite_score"]
    )

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "weight"]
        )

    n = max(
        1,
        int(np.ceil(
            len(df) * TOP_QUANTILE
        ))
    )

    selected = (
        df.sort_values(
            "composite_score",
            ascending=False
        )
        .head(n)
        .copy()
    )

    selected["weight"] = (
        1.0 / len(selected)
    )

    return selected[
        ["ticker", "weight"]
    ].reset_index(drop=True)


# ============================================================
# Portfolio return
# ============================================================

def calculate_period_return(
    portfolio,
    price_matrix,
    start_date,
    end_date
):
    """
    Calculate the holding-period return for a portfolio.

    The portfolio is formed using information available at
    start_date. To avoid assuming that we can trade at the same
    closing price used to generate the signal, performance begins
    on the first available trading date AFTER start_date.

    Return:
        first trading day after start_date
        -> end_date
    """

    if portfolio.empty:
        return np.nan

    tickers = portfolio["ticker"].tolist()

    available = [
        ticker
        for ticker in tickers
        if ticker in price_matrix.columns
    ]

    if not available:
        return np.nan

    # --------------------------------------------------------
    # Find the first tradable date after the signal date.
    # --------------------------------------------------------

    available_dates = price_matrix.index

    future_dates = available_dates[
        available_dates > start_date
    ]

    if len(future_dates) == 0:
        return np.nan

    execution_date = future_dates[0]

    # Make sure the ending date exists.
    if end_date not in available_dates:
        return np.nan

    if execution_date >= end_date:
        return np.nan

    try:
        start_prices = price_matrix.loc[
            execution_date,
            available
        ]

        end_prices = price_matrix.loc[
            end_date,
            available
        ]

    except KeyError:
        return np.nan

    returns = (
        end_prices / start_prices
    ) - 1.0

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if returns.empty:
        return np.nan

    weights = (
        portfolio
        .set_index("ticker")
        .loc[returns.index, "weight"]
    )

    # Renormalize if some holdings lack prices.
    weight_sum = weights.sum()

    if weight_sum <= 0:
        return np.nan

    weights = weights / weight_sum

    return float(
        (returns * weights).sum()
    )


# ============================================================
# Turnover
# ============================================================

def calculate_turnover(
    previous_portfolio,
    current_portfolio
):
    """
    One-way turnover is:

        0.5 * sum(|w_new - w_old|)

    This is the standard one-way portfolio turnover
    convention.
    """

    previous = (
        previous_portfolio
        .set_index("ticker")["weight"]
        if not previous_portfolio.empty
        else pd.Series(dtype=float)
    )

    current = (
        current_portfolio
        .set_index("ticker")["weight"]
        if not current_portfolio.empty
        else pd.Series(dtype=float)
    )

    all_tickers = previous.index.union(
        current.index
    )

    previous = previous.reindex(
        all_tickers,
        fill_value=0.0
    )

    current = current.reindex(
        all_tickers,
        fill_value=0.0
    )

    turnover = 0.5 * np.abs(
        current - previous
    ).sum()

    return float(turnover)


# ============================================================
# Performance metrics
# ============================================================

def calculate_max_drawdown(returns):
    if len(returns) == 0:
        return np.nan

    wealth = (1.0 + returns).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max
    ) - 1.0

    return float(drawdown.min())


def calculate_metrics(
    returns,
    start_date=None,
    end_date=None,
):
    """
    Calculate performance metrics from periodic portfolio returns.

    The backtest returns are monthly returns, so:
        - CAGR is annualized using 12 periods/year.
        - Volatility is annualized using sqrt(12).
        - Sharpe is annualized using sqrt(12).

    Returns are expected as decimal returns:
        0.05 = +5%
        -0.02 = -2%
    """

    returns = pd.Series(returns).dropna().astype(float)

    if returns.empty:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Total Return": np.nan,
        }

    # --------------------------------------------------------
    # Total compounded return
    # --------------------------------------------------------

    cumulative_growth = (1.0 + returns).prod()

    total_return = cumulative_growth - 1.0

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    periods_per_year = 12.0

    if (
        start_date is not None
        and end_date is not None
    ):
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)

        years = (
            end_timestamp - start_timestamp
        ).days / 365.25
    else:
        years = len(returns) / periods_per_year

    if years > 0 and cumulative_growth > 0:
        cagr = cumulative_growth ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    # --------------------------------------------------------
    # Annualized volatility
    # --------------------------------------------------------

    if len(returns) > 1:
        volatility = (
            returns.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
    else:
        volatility = np.nan

    # --------------------------------------------------------
    # Annualized Sharpe ratio
    #
    # Assuming zero risk-free rate.
    # --------------------------------------------------------

    if (
        len(returns) > 1
        and returns.std(ddof=1) > 0
    ):
        sharpe = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
    else:
        sharpe = np.nan

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    max_drawdown = calculate_max_drawdown(returns)

    return {
        "CAGR": float(cagr),
        "Volatility": float(volatility),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(max_drawdown),
        "Total Return": float(total_return),
    }

def run_backtest():
    print()
    print("Multi-Factor Backtest")
    print("=====================")
    print(
        f"Momentum weight: "
        f"{MOMENTUM_WEIGHT:.0%}"
    )
    print(
        f"Low-vol weight: "
        f"{LOW_VOL_WEIGHT:.0%}"
    )
    print(
        f"Top quantile: "
        f"{TOP_QUANTILE:.0%}"
    )
    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST_BPS:.1f} bps"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    signals = load_signal_data()
    prices = load_prices()

    price_matrix = get_price_matrix(
        prices
    )

    common_dates = sorted(
        set(signals["Date"])
        & set(price_matrix.index)
    )

    if len(common_dates) < 2:
        raise ValueError(
            "Not enough common signal/price dates."
        )

    # --------------------------------------------------------
    # Monthly rebalancing
    # --------------------------------------------------------
    # The signal dataset contains daily observations, but this
    # strategy is intended to rebalance once per month.
    #
    # For each calendar month, select the LAST available common
    # trading date. The portfolio formed on that date is then
    # held until the next monthly rebalance date.
    # --------------------------------------------------------

    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    common_dates_df = pd.DataFrame({
        "Date": common_dates
    })

    common_dates_df["Month"] = (
        common_dates_df["Date"].dt.to_period("M")
    )

    rebalance_dates = (
        common_dates_df
        .groupby("Month")["Date"]
        .max()
        .sort_values()
        .tolist()
    )

    print()
    print(
        f"Rebalance dates: "
        f"{len(rebalance_dates)}"
    )

    monthly_records = []
    holdings_records = []

    previous_portfolio = pd.DataFrame(
        columns=["ticker", "weight"]
    )

    gross_returns = []
    net_returns = []

    turnovers = []
    transaction_costs = []

    for i in range(
        len(rebalance_dates) - 1
    ):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]

        print(
            f"Running rebalance: "
            f"{start_date.date()} "
            f"-> "
            f"{end_date.date()}"
        )

        signal_slice = signals[
            signals["Date"] == start_date
        ].copy()

        portfolio = construct_portfolio(
            signal_slice
        )

        if portfolio.empty:
            continue

        # Save holdings.
        for _, row in portfolio.iterrows():
            holdings_records.append({
                "Date": start_date,
                "ticker": row["ticker"],
                "weight": row["weight"],
            })

        turnover = calculate_turnover(
            previous_portfolio,
            portfolio
        )

        transaction_cost = (
            turnover
            * TRANSACTION_COST_BPS
            / 10000.0
        )

        gross_return = (
            calculate_period_return(
                portfolio=portfolio,
                price_matrix=price_matrix,
                start_date=start_date,
                end_date=end_date,
            )
        )

        if pd.isna(gross_return):
            previous_portfolio = portfolio
            continue

        net_return = (
            gross_return
            - transaction_cost
        )

        gross_returns.append(
            gross_return
        )

        net_returns.append(
            net_return
        )

        turnovers.append(
            turnover
        )

        transaction_costs.append(
            transaction_cost
        )

        monthly_records.append({
            "Start Date": start_date,
            "End Date": end_date,
            "Gross Return": gross_return,
            "Turnover": turnover,
            "Transaction Cost": transaction_cost,
            "Net Return": net_return,
            "Holdings": len(portfolio),
        })

        previous_portfolio = portfolio

    monthly = pd.DataFrame(
        monthly_records
    )

    if monthly.empty:
        raise ValueError(
            "Backtest produced no return periods."
        )

    gross_returns = pd.Series(
        gross_returns
    )

    net_returns = pd.Series(
        net_returns
    )

    gross_metrics = calculate_metrics(
        gross_returns,
        start_date=monthly["Start Date"].iloc[0],
        end_date=monthly["End Date"].iloc[-1],
    )

    net_metrics = calculate_metrics(
        net_returns,
        start_date=monthly["Start Date"].iloc[0],
        end_date=monthly["End Date"].iloc[-1],
    )

    average_monthly_turnover = (
        monthly["Turnover"].mean()
    )

    annualized_turnover = (
        average_monthly_turnover
        * 12.0
    )

    total_transaction_cost = (
        monthly["Transaction Cost"].sum()
    )

    total_transaction_cost_bps = (
        total_transaction_cost
        * 10000.0
    )

    average_holdings = (
        monthly["Holdings"].mean()
    )

    summary = pd.DataFrame([{
        "Strategy":
            "Momentum + Low Volatility",

        "Momentum Weight":
            MOMENTUM_WEIGHT,

        "Low Volatility Weight":
            LOW_VOL_WEIGHT,

        "Transaction Cost (bps)":
            TRANSACTION_COST_BPS,

        "Gross CAGR":
            gross_metrics["CAGR"],

        "Gross Volatility":
            gross_metrics["Volatility"],

        "Gross Sharpe":
            gross_metrics["Sharpe"],

        "Gross Max Drawdown":
            gross_metrics["Max Drawdown"],

        "Gross Total Return":
            gross_metrics["Total Return"],

        "Net CAGR":
            net_metrics["CAGR"],

        "Net Volatility":
            net_metrics["Volatility"],

        "Net Sharpe":
            net_metrics["Sharpe"],

        "Net Max Drawdown":
            net_metrics["Max Drawdown"],

        "Net Total Return":
            net_metrics["Total Return"],

        "Average Monthly Turnover":
            average_monthly_turnover,

        "Annualized Turnover":
            annualized_turnover,

        "Total Transaction Cost":
            total_transaction_cost,

        "Total Transaction Cost (bps)":
            total_transaction_cost_bps,

        "Average Holdings":
            average_holdings,

        "Rebalance Periods":
            len(monthly),
    }])

    # Save files.
    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    monthly.to_csv(
        MONTHLY_FILE,
        index=False
    )

    holdings = pd.DataFrame(
        holdings_records
    )

    holdings.to_csv(
        HOLDINGS_FILE,
        index=False
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("Multi-Factor Results")
    print("====================")

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved summary to: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Saved monthly results to: "
        f"{MONTHLY_FILE}"
    )

    print(
        f"Saved holdings to: "
        f"{HOLDINGS_FILE}"
    )


if __name__ == "__main__":
    run_backtest()
