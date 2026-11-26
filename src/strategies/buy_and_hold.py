from src.strategies.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Buy on day one, hold forever."""

    LEARNS = False

    def train(self, data):
        pass

    def get_action(self, row) -> str:
        return 'buy'

    def reset(self):
        pass
