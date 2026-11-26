import pandas as pd
from src.strategies.base import Strategy


class SMAStrategy(Strategy):
    """
    Simple Moving Average crossover strategy.
    Buy when short SMA crosses above long SMA, sell when it crosses below.

    SMA has no learnable parameters, so LEARNS = False. Validators should pass
    the full window (train+test) to train() so test-period signals are available.
    """

    LEARNS = False

    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window
        self.reset()

    def train(self, data):
        prices = data['spy_close']
        short_sma = prices.rolling(self.short_window).mean()
        long_sma = prices.rolling(self.long_window).mean()

        signals = {}
        for i in range(len(data)):
            date = data.index[i]
            if pd.isna(short_sma.iloc[i]) or pd.isna(long_sma.iloc[i]):
                signals[date] = 'hold'
            elif short_sma.iloc[i] > long_sma.iloc[i]:
                signals[date] = 'buy'
            else:
                signals[date] = 'sell'

        self.signals = signals

    def get_action(self, row) -> str:
        return self.signals.get(row.Index, 'hold')

    def reset(self):
        self.signals = {}
