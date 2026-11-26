import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from src.validation.base import Validator
from src.backtest import simulate, compute_metrics, INITIAL_CAPITAL


class RollingWindowValidator(Validator):
    """
    Walk-forward validation: trains on a rolling window, tests on the next period.

    Note: regime labels and feature bins are computed once on the full dataset
    before this validator is called. Only the strategy parameters (Q-table) are
    re-fitted per window. For a fully rigorous study the HMM would also be
    re-fitted per window.
    """

    def __init__(
        self,
        train_years: int = 5,
        test_months: int = 12,
        step_months: int = 12,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.train_years = train_years
        self.test_months = test_months
        self.step_months = step_months
        self.initial_capital = initial_capital

    def _windows(self, df):
        """Generate (train_start, train_end, test_end) date triples."""
        start = df.index.min()
        end = df.index.max()

        train_start = start
        while True:
            train_end = train_start + relativedelta(years=self.train_years)
            test_end = train_end + relativedelta(months=self.test_months)
            if test_end > end:
                break
            yield train_start, train_end, test_end
            train_start += relativedelta(months=self.step_months)

    def run(self, df, strategies: list) -> dict:
        """
        strategies: list of (name, strategy_instance) pairs.
        Returns {name: {'windows': [...], 'summary': {...}}}.
        """
        windows = list(self._windows(df))
        if not windows:
            raise ValueError("No complete windows fit in the data with the given parameters.")

        all_results = {name: [] for name, _ in strategies}

        for win_idx, (train_start, train_end, test_end) in enumerate(windows):
            train_df = df[(df.index >= train_start) & (df.index < train_end)]
            test_df  = df[(df.index >= train_end)   & (df.index < test_end)]
            full_window_df = df[(df.index >= train_start) & (df.index < test_end)]

            if len(train_df) < 50 or len(test_df) < 5:
                continue

            print(f"  Window {win_idx + 1}/{len(windows)}: "
                  f"train {train_start.date()} → {train_end.date()} | "
                  f"test {train_end.date()} → {test_end.date()} "
                  f"({len(train_df)} / {len(test_df)} days)")

            for name, strategy in strategies:
                strategy.reset()
                if strategy.LEARNS:
                    strategy.train(train_df)
                else:
                    strategy.train(full_window_df)

                trades, values = simulate(test_df, strategy, self.initial_capital)
                metrics = compute_metrics(values, trades, self.initial_capital)
                metrics['train_start'] = train_start
                metrics['train_end'] = train_end
                metrics['test_start'] = train_end
                metrics['test_end'] = test_end
                all_results[name].append(metrics)

        return {
            name: {
                'windows': all_results[name],
                'summary': self._summarize(name, all_results[name], all_results),
            }
            for name, _ in strategies
        }

    def _summarize(self, name, window_results, all_results):
        if not window_results:
            return {}

        returns   = [w['total_return_pct'] for w in window_results]
        sharpes   = [w['sharpe'] for w in window_results]
        sortinos  = [w['sortino'] for w in window_results]
        drawdowns = [w['max_drawdown_pct'] for w in window_results]
        cagrs     = [w['cagr_pct'] for w in window_results]

        # Win rate vs buy & hold (if present)
        bnh_returns = None
        if 'Buy & Hold' in all_results and all_results['Buy & Hold']:
            bnh_returns = [w['total_return_pct'] for w in all_results['Buy & Hold']]

        win_rate = None
        if bnh_returns and name != 'Buy & Hold':
            n = min(len(returns), len(bnh_returns))
            wins = sum(r > b for r, b in zip(returns[:n], bnh_returns[:n]))
            win_rate = wins / n * 100

        return {
            'mean_total_return_pct': float(np.mean(returns)),
            'std_total_return_pct':  float(np.std(returns)),
            'mean_cagr_pct':         float(np.mean(cagrs)),
            'mean_sharpe':           float(np.mean(sharpes)),
            'mean_sortino':          float(np.mean(sortinos)),
            'mean_max_drawdown_pct': float(np.mean(drawdowns)),
            'n_windows':             len(window_results),
            'win_rate_vs_bnh_pct':   win_rate,
        }
