# Q-Learning Trading Validation

A research project validating whether a Q-learning agent can outperform simple baselines (Buy & Hold, SMA crossover) on SPY using HMM-detected market regimes.

## Architecture

```
src/
  data.py              — SPY + VIX loading via yfinance
  regime_detection.py  — GaussianHMM: fit on train, predict for test (no leakage)
  features.py          — Discretize log returns + VIX into 3 bins each (27-state space)
  metrics.py           — Sharpe, Sortino, max drawdown, CAGR
  backtest.py          — simulate() and compute_metrics() shared by all scripts

  strategies/
    base.py            — Abstract Strategy (LEARNS flag + get_action interface)
    q_learning.py      — Tabular Q-learning, epsilon-greedy, 1000 episodes
    sma.py             — SMA 50/200 crossover
    buy_and_hold.py    — Perpetual buy baseline

  validation/
    base.py            — Abstract Validator
    full_period.py     — Single train/test split
    rolling_window.py  — Walk-forward validation (rolling train window)

scripts/
  run_experiment.py    — Single-run experiment (2000–2019 train, 2019–present test)
  compare_results.py   — Rolling walk-forward comparison across all strategies
```

## State Space

The Q-table has **27 states**: `regime × return_bin × vix_bin`

| Dimension     | Values                      |
|---------------|-----------------------------|
| Regime (HMM)  | bull / sideways / bear      |
| Log return    | low / medium / high (33rd/67th pct) |
| VIX           | low / neutral / high (33rd/67th pct) |

Actions: `buy` / `sell` / `hold`  
Reward: excess return over market (portfolio return − SPY daily return)

## Data Integrity

**No look-ahead bias:**
- HMM fitted exclusively on the training period
- Bin edges (return/VIX percentiles) computed from training data, applied to test data
- Rolling validation re-trains the Q-table per window; regime labels are fixed (see note below)

> **Note:** For a fully rigorous walk-forward study, the HMM would be re-fitted for each rolling window. Currently, regime labels are computed once on the full pre-cutoff dataset. This is a known simplification.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Single experiment (2000–2019 train, 2019–present test)
python -m scripts.run_experiment

# Rolling walk-forward comparison (5-year train, 1-year test, 2000–2019)
python -m scripts.compare_results
```

## Metrics Reported

| Metric       | Description                              |
|--------------|------------------------------------------|
| Total return | Cumulative % gain over test period       |
| CAGR         | Compound annual growth rate              |
| Sharpe       | Annualised return / annualised std dev. Higher is better; >1 is generally considered good |
| Sortino      | Like Sharpe but penalises only downside volatility — better for asymmetric return distributions |
| Max drawdown | Largest peak-to-trough % decline. Measures worst-case loss from a high-water mark |
| Win rate     | % of rolling windows beating Buy & Hold |

## Results (test period: 2019–2026)

Train: 2000–2018 (4,778 days) | Test: 2019–present (1,987 days)

| Metric        | Q-Learning | Buy & Hold | SMA (50/200) |
|---------------|-----------|------------|--------------|
| Total return  | **224.87%** | 217.02%  | 104.48%      |
| CAGR          | **16.1%**   | 15.8%    | 9.5%         |
| Sharpe        | **0.879**   | 0.862    | 0.645        |
| Sortino       | **1.081**   | 1.062    | 0.684        |
| Max drawdown  | -33.31%     | -33.31%  | -33.26%      |
| Trades        | 13          | 1        | 7            |

### Key observations

**Q-Learning marginally beats Buy & Hold** (+2.48% on final value, slightly better Sharpe and Sortino), but with only 13 trades over ~8 years the gap is small enough to be noise from individual trade timing.

**Identical max drawdown** between Q-Learning and Buy & Hold (−33.31%) confirms the agent learned to stay long almost continuously — it converged toward Buy & Hold behaviour rather than discovering a genuinely different strategy.

**Training converged correctly:** average reward went from −2.0 (random, overtrades) to +0.05 (selective). Trades per episode collapsed from ~1,500 → ~90, showing the agent learned that transaction costs punish churn. This is correct RL behaviour, but the 27-state table may be too coarse to find real alpha beyond "don't overtrade."

**SMA significantly underperformed** (104% vs 217%) because the 2019–2026 period was a prolonged bull run — any strategy that sells gets left behind.

**Next step:** run `compare_results.py` for rolling walk-forward validation across different market regimes to test whether the marginal edge holds out of sample.
