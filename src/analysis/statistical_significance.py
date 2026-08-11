"""
Statistical significance analysis for the multi-factor strategy.

Tests whether the strategy's monthly returns are statistically
different from zero using a one-sample t-test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "multi_factor_monthly.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "statistical_significance.csv"


def main():
    print("Statistical Significance Analysis")
    print("=" * 35)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded: {INPUT_FILE}")

    # Identify the strategy return column.
    possible_columns = [
        "net_return",
        "Net Return",
        "net_monthly_return",
        "strategy_return",
        "return",
    ]

    return_column = next(
        (column for column in possible_columns if column in df.columns),
        None,
    )

    if return_column is None:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError(
            "Could not identify the monthly strategy return column."
        )

    returns = pd.to_numeric(df[return_column], errors="coerce").dropna()

    if len(returns) < 2:
        raise ValueError("Not enough observations for statistical testing.")

    # One-sample t-test:
    # H0: mean monthly return = 0
    # H1: mean monthly return != 0
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)

    n = len(returns)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    standard_error = std_return / np.sqrt(n)

    # 95% confidence interval for the mean monthly return.
    critical_value = stats.t.ppf(0.975, df=n - 1)

    ci_low = mean_return - critical_value * standard_error
    ci_high = mean_return + critical_value * standard_error

    # Annualized Sharpe ratio using monthly observations.
    sharpe = (mean_return / std_return) * np.sqrt(12)

    # Annualized mean return approximation.
    annualized_mean = mean_return * 12

    results = pd.DataFrame(
        {
            "metric": [
                "observations",
                "mean_monthly_return",
                "annualized_mean_return",
                "monthly_volatility",
                "annualized_sharpe",
                "t_statistic",
                "p_value",
                "confidence_interval_95_low",
                "confidence_interval_95_high",
            ],
            "value": [
                n,
                mean_return,
                annualized_mean,
                std_return,
                sharpe,
                t_stat,
                p_value,
                ci_low,
                ci_high,
            ],
        }
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nResults")
    print("-------")
    print(f"Observations:           {n}")
    print(f"Mean monthly return:    {mean_return:.6f}")
    print(f"Annualized mean return: {annualized_mean:.6f}")
    print(f"Monthly volatility:     {std_return:.6f}")
    print(f"Annualized Sharpe:      {sharpe:.4f}")
    print(f"t-statistic:            {t_stat:.4f}")
    print(f"p-value:                {p_value:.6f}")
    print(f"95% CI:                 [{ci_low:.6f}, {ci_high:.6f}]")

    print("\nHypothesis")
    print("----------")
    print("H0: Mean monthly return = 0")
    print("H1: Mean monthly return != 0")

    if p_value < 0.05:
        print("Result: Reject H0 at the 5% significance level.")
    else:
        print("Result: Fail to reject H0 at the 5% significance level.")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()