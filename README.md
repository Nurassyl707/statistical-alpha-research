# Statistical Alpha Research & Walk-Forward Backtesting Engine

A quantitative research pipeline for evaluating cross-sectional equity factor
strategies using historical U.S. equity data.

The project investigates whether a systematic combination of **multi-horizon
momentum** and **low-volatility** signals can generate persistent
risk-adjusted returns after portfolio construction, turnover, and transaction
costs.

The research pipeline includes:

- Cross-sectional factor construction
- Multi-horizon momentum signals
- Low-volatility signals
- Factor ranking and composite alpha construction
- Monthly walk-forward portfolio backtesting
- Transaction-cost modeling
- Portfolio turnover analysis
- Benchmark comparison against SPY
- Factor attribution
- Robustness analysis
- Leave-one-out analysis
- Factor IC correlation analysis
- Independent backtest validation
- Statistical significance testing

---

## Research Objective

The objective is to investigate whether combining momentum and low-volatility
signals can produce a portfolio with attractive risk-adjusted returns across
a broad U.S. equity universe.

The main strategy combines:

- **70% momentum**
- **30% low volatility**
- **Top 10% of ranked stocks**
- **Monthly rebalancing**
- **10 bps transaction-cost assumption**

The backtest uses a walk-forward design so that portfolio decisions at each
rebalance date are based only on information available at that point in time.

---

## Methodology

### Universe

- S&P 500 equity universe
- Approximately 500 stocks
- Daily adjusted closing prices
- Historical period: 2020–2026
- Signal formation begins after sufficient historical observations are
  available

### Momentum Factors

Momentum is evaluated across multiple horizons:

- 21 trading days
- 63 trading days
- 126 trading days
- 252 trading days

Each momentum signal is converted into a cross-sectional rank.

A composite momentum signal is then constructed from the available momentum
horizons.

### Low-Volatility Factor

Volatility is measured using rolling historical return volatility over:

- 21 trading days
- 63 trading days
- 252 trading days

Lower-volatility stocks receive higher low-volatility scores.

### Alpha Construction

The primary multi-factor signal combines momentum and low volatility:

```text
Alpha Score =
    70% × Momentum Score
  + 30% × Low-Volatility Score

  ## Statistical Significance

The final strategy's monthly returns were tested against the null hypothesis that the mean monthly return is zero.

| Statistic | Result |
|---|---:|
| Observations | 68 |
| Mean Monthly Return | 0.8783% |
| Annualized Mean Return | 10.54% |
| Monthly Volatility | 3.85% |
| Annualized Sharpe | 0.79 |
| t-statistic | 1.88 |
| p-value | 0.0641 |
| 95% Confidence Interval | [-0.053%, 1.810%] |

### Hypothesis Test

- **H₀:** Mean monthly return = 0
- **H₁:** Mean monthly return ≠ 0

The p-value is **0.0641**. Therefore, the null hypothesis is not rejected at the conventional 5% significance level.

This means that while the strategy demonstrates positive historical performance, the evidence is not strong enough at the 5% level to conclude that its mean monthly return is statistically different from zero.

The analysis can be reproduced with:

```bash
python src/analysis/statistical_significance.py