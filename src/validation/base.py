from abc import ABC, abstractmethod


class Validator(ABC):
    @abstractmethod
    def run(self, df, strategies: list) -> dict:
        """
        Validate a list of (name, strategy_instance) pairs against df.
        Returns a results dict keyed by strategy name.
        """
        pass
