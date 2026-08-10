from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


BACKTEST_FILE = Path("data/processed/backtest_returns.csv")
OUTPUT_FILE = Path("data/processed/benchmark_comparison.csv")


START_DATE = "2021-01-01"
END_DATE = "2026-08-08"


def load_strategy_returns():
    """
    Load strategy daily returns.
    """

    if not BACKTEST_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {BACKTEST_FILE}"
        )

    df = pd.read_csv(BACKTEST_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        ["Date", "portfolio_return"]
    ].copy()

    df = df.sort_values("Date")

    df["portfolio_return"] = pd.to_numeric(
        df["portfolio_return"],
        errors="coerce"
    )

    if df["portfolio_return"].isna().any():
        raise ValueError(
            "Strategy contains missing returns."
        )

    return df.reset_index(drop=True)


def download_spy_returns():
    """
    Download SPY adjusted prices and calculate daily returns.
    """

    print("Downloading SPY benchmark data...")

    spy = yf.download(
        "SPY",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if spy.empty:
        raise RuntimeError(
            "Failed to download SPY data."
        )

    # Handle yfinance MultiIndex columns.
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    if "Adj Close" not in spy.columns:
        raise ValueError(
            "SPY data does not contain Adj Close."
        )

    spy = spy[
        ["Adj Close"]
    ].copy()

    spy.index = pd.to_datetime(
        spy.index
    )

    spy = spy.rename(
        columns={
            "Adj Close": "SPY"
        }
    )

    spy["spy_return"] = (
        spy["SPY"]
        .pct_change()
    )

    spy = spy.dropna()

    spy = spy.reset_index()

    spy = spy.rename(
        columns={
            "Date": "Date"
        }
    )

    return spy[
        ["Date", "spy_return"]
    ]


def calculate_performance(returns):
    """
    Calculate standard performance statistics.
    """

    returns = returns.dropna()

    cumulative = (
        1 + returns
    ).cumprod()

    final_wealth = cumulative.iloc[-1]

    years = len(returns) / 252

    annualized_return = (
        final_wealth ** (1 / years)
    ) - 1

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(252)
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    maximum_drawdown = drawdown.min()

    return {
        "final_wealth": final_wealth,
        "cumulative_return": final_wealth - 1,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def calculate_comparison(strategy, benchmark):
    """
    Compare strategy against benchmark.
    """

    merged = pd.merge(
        strategy,
        benchmark,
        on="Date",
        how="inner"
    )

    if merged.empty:
        raise RuntimeError(
            "No overlapping dates between "
            "strategy and benchmark."
        )

    strategy_returns = (
        merged["portfolio_return"]
    )

    benchmark_returns = (
        merged["spy_return"]
    )

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    strategy_metrics = calculate_performance(
        strategy_returns
    )

    benchmark_metrics = calculate_performance(
        benchmark_returns
    )

    # ---------------------------------------------------------
    # Excess returns
    # ---------------------------------------------------------

    excess_returns = (
        strategy_returns
        - benchmark_returns
    )

    information_ratio = (
        excess_returns.mean()
        / excess_returns.std(ddof=1)
        * np.sqrt(252)
    )

    tracking_error = (
        excess_returns.std(ddof=1)
        * np.sqrt(252)
    )

    # ---------------------------------------------------------
    # Beta
    # ---------------------------------------------------------

    covariance = np.cov(
        strategy_returns,
        benchmark_returns,
        ddof=1
    )[0, 1]

    benchmark_variance = np.var(
        benchmark_returns,
        ddof=1
    )

    beta = (
        covariance
        / benchmark_variance
    )

    # ---------------------------------------------------------
    # Jensen's alpha
    #
    # For now assume risk-free rate = 0.
    # ---------------------------------------------------------

    annualized_strategy_return = (
        strategy_metrics["annualized_return"]
    )

    annualized_benchmark_return = (
        benchmark_metrics["annualized_return"]
    )

    jensen_alpha = (
        annualized_strategy_return
        - beta * annualized_benchmark_return
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results = {
        "strategy_cumulative_return":
            strategy_metrics["cumulative_return"],

        "benchmark_cumulative_return":
            benchmark_metrics["cumulative_return"],

        "strategy_annualized_return":
            strategy_metrics["annualized_return"],

        "benchmark_annualized_return":
            benchmark_metrics["annualized_return"],

        "strategy_annualized_volatility":
            strategy_metrics["annualized_volatility"],

        "benchmark_annualized_volatility":
            benchmark_metrics["annualized_volatility"],

        "strategy_sharpe":
            strategy_metrics["sharpe"],

        "benchmark_sharpe":
            benchmark_metrics["sharpe"],

        "strategy_maximum_drawdown":
            strategy_metrics["maximum_drawdown"],

        "benchmark_maximum_drawdown":
            benchmark_metrics["maximum_drawdown"],

        "beta":
            beta,

        "jensen_alpha":
            jensen_alpha,

        "tracking_error":
            tracking_error,

        "information_ratio":
            information_ratio,

        "excess_cumulative_return":
            strategy_metrics["cumulative_return"]
            - benchmark_metrics["cumulative_return"],
    }

    return merged, results


def save_results(results):
    """
    Save benchmark comparison results.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        [results]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


def print_results(results):
    """
    Print benchmark comparison.
    """

    print()
    print("Benchmark Comparison")
    print("====================")

    print(
        f"Strategy cumulative return:   "
        f"{results['strategy_cumulative_return']:.2%}"
    )

    print(
        f"SPY cumulative return:        "
        f"{results['benchmark_cumulative_return']:.2%}"
    )

    print()

    print(
        f"Strategy annualized return:   "
        f"{results['strategy_annualized_return']:.2%}"
    )

    print(
        f"SPY annualized return:        "
        f"{results['benchmark_annualized_return']:.2%}"
    )

    print()

    print(
        f"Strategy volatility:          "
        f"{results['strategy_annualized_volatility']:.2%}"
    )

    print(
        f"SPY volatility:               "
        f"{results['benchmark_annualized_volatility']:.2%}"
    )

    print()

    print(
        f"Strategy Sharpe:              "
        f"{results['strategy_sharpe']:.4f}"
    )

    print(
        f"SPY Sharpe:                   "
        f"{results['benchmark_sharpe']:.4f}"
    )

    print()

    print(
        f"Strategy max drawdown:        "
        f"{results['strategy_maximum_drawdown']:.2%}"
    )

    print(
        f"SPY max drawdown:             "
        f"{results['benchmark_maximum_drawdown']:.2%}"
    )

    print()

    print(
        f"Beta:                         "
        f"{results['beta']:.4f}"
    )

    print(
        f"Jensen alpha:                 "
        f"{results['jensen_alpha']:.2%}"
    )

    print(
        f"Tracking error:               "
        f"{results['tracking_error']:.2%}"
    )

    print(
        f"Information ratio:            "
        f"{results['information_ratio']:.4f}"
    )

    print(
        f"Excess cumulative return:     "
        f"{results['excess_cumulative_return']:.2%}"
    )

    print()

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


def run_benchmark_analysis():
    """
    Complete benchmark analysis.
    """

    strategy = load_strategy_returns()

    benchmark = download_spy_returns()

    merged, results = calculate_comparison(
        strategy,
        benchmark
    )

    save_results(results)

    print_results(results)

    return merged, results


if __name__ == "__main__":
    run_benchmark_analysis()