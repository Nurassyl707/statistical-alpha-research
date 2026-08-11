# Statistical Alpha Research & Walk-Forward Backtesting Engine

A quantitative research pipeline for evaluating cross-sectional equity factor strategies using historical U.S. equity data.

The project investigates whether a systematic combination of **multi-horizon momentum** and **low-volatility** signals can generate persistent risk-adjusted returns across a broad equity universe.

The pipeline covers:

- Historical market-data processing
- Feature engineering
- Cross-sectional factor ranking
- Multi-factor alpha construction
- Portfolio construction
- Monthly walk-forward backtesting
- Transaction-cost modeling
- Turnover analysis
- Benchmark comparison
- Factor attribution
- Factor correlation analysis
- Robustness testing
- Leave-one-out analysis
- Independent backtest validation

---

## Research Objective

The objective is to evaluate whether combining momentum and low-volatility factors produces a portfolio with attractive risk-adjusted performance relative to a broad market benchmark.

The primary strategy combines:

- **70% momentum**
- **30% low volatility**
- **Top 10% of ranked stocks**
- **Monthly rebalancing**
- **10 bps transaction cost assumption**

The research is designed around an out-of-sample-style walk-forward framework rather than a single full-period optimization.

---

## Methodology

### 1. Investment Universe

The research uses approximately 500 U.S. equities corresponding to the S&P 500 universe.

Historical daily adjusted closing prices are used to construct the factor signals.

Sample period:

```text
2020-01-02 → 2026-08-07