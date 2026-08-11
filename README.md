# Statistical Alpha Research & Walk-Forward Backtesting Engine

A quantitative research pipeline for evaluating cross-sectional equity factor strategies using historical S&P 500 data.

The project combines multi-horizon momentum and low-volatility signals, constructs portfolios based on cross-sectional rankings, and evaluates performance through a monthly walk-forward backtest with transaction costs.

## Research Objective

The objective is to investigate whether a systematic combination of momentum and low-volatility factors can generate persistent risk-adjusted returns across a broad U.S. equity universe.

## Methodology

### Universe

- S&P 500 constituents
- 500+ stocks
- Historical price data from 2020–2026
- Daily adjusted closing prices

### Features

The research pipeline constructs:

- 1-day returns
- 5-day returns
- 21-day returns
- 63-day returns
- 126-day returns
- 252-day returns
- 21-day volatility
- 63-day volatility
- 252-day volatility

### Factor Construction

Momentum is constructed from multiple lookback horizons and ranked cross-sectionally by date.

Low volatility is constructed by ranking stocks according to historical realized volatility.

The primary alpha model combines:

- 70% Momentum
- 30% Low Volatility

Stocks are ranked using the resulting composite alpha score.

### Portfolio Construction

The strategy:

1. Ranks the cross-section of stocks at each rebalance date.
2. Selects the highest-ranked stocks.
3. Constructs an approximately equal-weighted portfolio.
4. Rebalances monthly.
5. Applies transaction costs.

Transaction costs are modeled at:

**10 basis points per unit of turnover.**

## Walk-Forward Backtest

The backtest uses monthly rebalancing and evaluates the strategy from December 2020 through August 2026.

The backtest explicitly accounts for:

- Portfolio weights
- Rebalancing
- Turnover
- Transaction costs
- Missing price observations
- Portfolio returns
- Drawdowns
- Risk-adjusted performance

## Results

### Multi-Factor Strategy

| Metric | Result |
|---|---:|
| Gross CAGR | 10.84% |
| Net CAGR | 10.24% |
| Gross Sharpe | 0.83 |
| Net Sharpe | 0.79 |
| Gross Max Drawdown | -14.42% |
| Net Max Drawdown | -14.75% |
| Gross Total Return | 77.91% |
| Net Total Return | 72.60% |
| Average Monthly Turnover | 44.97% |
| Annualized Turnover | 5.40× |
| Transaction Cost | 10 bps |
| Average Holdings | ~50 |
| Rebalance Periods | 68 |

### Benchmark Analysis

The strategy was also compared against SPY over the corresponding evaluation period.

| Metric | Strategy | SPY |
|---|---:|---:|
| Annualized Return | 19.53% | 15.74% |
| Volatility | 17.01% | 16.80% |
| Sharpe | 1.13 | 0.95 |
| Max Drawdown | -17.30% | -24.50% |

Additional benchmark statistics:

- Beta: 0.71
- Jensen's alpha: 8.35%
- Tracking error: 13.07%
- Information ratio: 0.25

## Robustness Analysis

The project evaluates alternative:

- Momentum lookback periods
- Portfolio sizes
- Factor combinations
- Factor correlations

Momentum strategies were tested using 3-month, 6-month, and 12-month lookback horizons with different portfolio sizes.

These experiments are used to examine whether observed performance depends heavily on a single parameter choice.

## Validation

The backtest includes automated validation of:

- Portfolio weight normalization
- Turnover calculations
- Transaction costs
- Total returns
- CAGR
- Sharpe ratio
- Maximum drawdown

The validation script independently recomputes reported performance metrics.

Example:

```text
Weight validation: PASS
Turnover validation: PASS
Transaction cost validation: PASS
Performance metric validation: PASS