from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_FILE = Path("data/processed/multi_factor_backtest.csv")
MONTHLY_FILE = Path("data/processed/multi_factor_monthly.csv")
HOLDINGS_FILE = Path("data/processed/multi_factor_holdings.csv")


def check_file(path):
    if not path.exists():
        print(f"FAIL: Missing file: {path}")
        return False

    print(f"PASS: Found {path}")
    return True


def check_holdings():
    print("\nWEIGHT CHECK")
    print("------------")

    df = pd.read_csv(HOLDINGS_FILE)

    weight_sums = df.groupby("Date")["weight"].sum()

    max_deviation = (weight_sums - 1.0).abs().max()

    print(f"Rebalance dates:        {len(weight_sums)}")
    print(f"Average weight sum:     {weight_sums.mean():.6f}")
    print(f"Minimum weight sum:     {weight_sums.min():.6f}")
    print(f"Maximum weight sum:     {weight_sums.max():.6f}")
    print(f"Max deviation from 1:   {max_deviation:.10f}")

    if max_deviation < 1e-8:
        print("PASS")
        return True

    print("FAIL")
    return False


def check_turnover_and_costs():
    print("\nTURNOVER / TRANSACTION COST CHECK")
    print("----------------------------------")

    df = pd.read_csv(MONTHLY_FILE)

    expected_cost = df["Turnover"] * 0.001

    max_cost_error = (
        expected_cost - df["Transaction Cost"]
    ).abs().max()

    print(f"Average monthly turnover: {df['Turnover'].mean():.4f}")
    print(f"Annualized turnover:       {df['Turnover'].mean() * 12:.4f}")
    print(f"Total reported TC:         {df['Transaction Cost'].sum():.6f}")
    print(f"Max TC calculation error:  {max_cost_error:.10f}")

    if max_cost_error < 1e-10:
        print("PASS")
        return True

    print("FAIL")
    return False


def calculate_total_return(returns):
    return (1.0 + returns).prod() - 1.0


def calculate_cagr(returns, start_date, end_date):
    total_return = calculate_total_return(returns)

    years = (
        pd.Timestamp(end_date) - pd.Timestamp(start_date)
    ).days / 365.25

    if years <= 0:
        return np.nan

    return (1.0 + total_return) ** (1.0 / years) - 1.0


def calculate_sharpe(returns):
    if returns.std(ddof=1) == 0:
        return np.nan

    return (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(12)
    )


def calculate_max_drawdown(returns):
    equity = (1.0 + returns).cumprod()

    running_max = equity.cummax()

    drawdown = equity / running_max - 1.0

    return drawdown.min()


def check_performance_metrics():
    print("\nPERFORMANCE METRIC VALIDATION")
    print("------------------------------")

    backtest = pd.read_csv(BACKTEST_FILE).iloc[0]
    monthly = pd.read_csv(MONTHLY_FILE)

    monthly["Start Date"] = pd.to_datetime(monthly["Start Date"])
    monthly["End Date"] = pd.to_datetime(monthly["End Date"])

    gross_returns = monthly["Gross Return"]
    net_returns = monthly["Net Return"]

    start_date = monthly["Start Date"].iloc[0]
    end_date = monthly["End Date"].iloc[-1]

    print(f"Return periods: {len(monthly)}")
    print(f"First period end: {start_date.date()}")
    print(f"Last period end:  {end_date.date()}")

    # -----------------------------
    # Gross
    # -----------------------------

    gross_total = calculate_total_return(gross_returns)
    gross_cagr = calculate_cagr(
        gross_returns,
        start_date,
        end_date,
    )
    gross_sharpe = calculate_sharpe(gross_returns)
    gross_mdd = calculate_max_drawdown(gross_returns)

    print("\nGross metrics")
    print(f"Reported total return: {backtest['Gross Total Return']:.6f}")
    print(f"Calculated total return: {gross_total:.6f}")

    print(f"Reported CAGR: {backtest['Gross CAGR']:.6f}")
    print(f"Calculated CAGR: {gross_cagr:.6f}")

    print(f"Reported Sharpe: {backtest['Gross Sharpe']:.6f}")
    print(f"Calculated Sharpe: {gross_sharpe:.6f}")

    print(f"Reported Max DD: {backtest['Gross Max Drawdown']:.6f}")
    print(f"Calculated Max DD: {gross_mdd:.6f}")

    # -----------------------------
    # Net
    # -----------------------------

    net_total = calculate_total_return(net_returns)
    net_cagr = calculate_cagr(
        net_returns,
        start_date,
        end_date,
    )
    net_sharpe = calculate_sharpe(net_returns)
    net_mdd = calculate_max_drawdown(net_returns)

    print("\nNet metrics")
    print(f"Reported total return: {backtest['Net Total Return']:.6f}")
    print(f"Calculated total return: {net_total:.6f}")

    print(f"Reported CAGR: {backtest['Net CAGR']:.6f}")
    print(f"Calculated CAGR: {net_cagr:.6f}")

    print(f"Reported Sharpe: {backtest['Net Sharpe']:.6f}")
    print(f"Calculated Sharpe: {net_sharpe:.6f}")

    print(f"Reported Max DD: {backtest['Net Max Drawdown']:.6f}")
    print(f"Calculated Max DD: {net_mdd:.6f}")

    print("\nMetric differences")

    print(
        f"CAGR difference: "
        f"{abs(backtest['Gross CAGR'] - gross_cagr):.10f}"
    )

    print(
        f"Sharpe difference: "
        f"{abs(backtest['Gross Sharpe'] - gross_sharpe):.10f}"
    )

    print(
        f"Max DD difference: "
        f"{abs(backtest['Gross Max Drawdown'] - gross_mdd):.10f}"
    )


def main():
    print("Backtest Validation")
    print("===================")

    print("\nFILE CHECK")
    print("----------")

    files_ok = all(
        check_file(path)
        for path in [
            BACKTEST_FILE,
            MONTHLY_FILE,
            HOLDINGS_FILE,
        ]
    )

    if not files_ok:
        print("\nValidation stopped because required files are missing.")
        return

    check_holdings()

    check_turnover_and_costs()

    check_performance_metrics()

    print("\nValidation complete.")


if __name__ == "__main__":
    main()

