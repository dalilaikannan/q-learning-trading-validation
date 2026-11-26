import pandas as pd
from src.validation.base import Validator
from src.backtest import simulate, compute_metrics, INITIAL_CAPITAL


class FullPeriodValidator(Validator):
    """Single train/test split validation."""

    def __init__(self, train_end: str, initial_capital: float = INITIAL_CAPITAL):
        self.train_end = pd.Timestamp(train_end)
        self.initial_capital = initial_capital

    def run(self, df, strategies: list) -> dict:
        """
        strategies: list of (name, strategy_instance) pairs.
        Returns {name: metrics_dict} for the test period.
        """
        train_df = df[df.index < self.train_end]
        test_df = df[df.index >= self.train_end]

        results = {}
        for name, strategy in strategies:
            strategy.reset()
            if strategy.LEARNS:
                strategy.train(train_df)
            else:
                strategy.train(df)  # SMA/BnH get full data for signal precomputation

            trades, values = simulate(test_df, strategy, self.initial_capital)
            results[name] = compute_metrics(values, trades, self.initial_capital)

        return results
