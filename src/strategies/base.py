from abc import ABC, abstractmethod


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    # Subclasses with learnable parameters set LEARNS = True.
    # Non-learning strategies (e.g. SMA) set LEARNS = False so validators
    # can pass them the full window (train+test) for signal precomputation.
    LEARNS = True

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def get_action(self, row) -> str:
        """Return 'buy', 'sell', or 'hold' for a named-tuple row from itertuples()."""
        pass

    @abstractmethod
    def reset(self):
        pass
