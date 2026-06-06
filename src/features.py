"""
Feature engineering for Q-learning state space & converts raw market data into discrete state representations
"""

import numpy as np
import pandas as pd


def discretize_returns(log_returns):
    """
    Discretize log returns into low/medium/high bins & uses percentiles so it works for any asset
    """
    log_returns = log_returns.squeeze()
    low = log_returns.quantile(0.33)
    high = log_returns.quantile(0.67)

    return pd.cut(
        log_returns,
        bins=[-np.inf, low, high, np.inf],
        labels=["low", "medium", "high"]
    )


def discretize_vix(vix):
    """
    Discretize VIX into low/neutral/high bins
    """
    vix = vix.squeeze()
    low = vix.quantile(0.33)
    high = vix.quantile(0.67)

    return pd.cut(
        vix,
        bins=[-np.inf, low, high, np.inf],
        labels=["low", "neutral", "high"]
    )


def build_state_space(log_returns, vix, regimes, stock_close, index):
    """
    Build the full state space DataFrame for Q-learning
    """
    log_returns_aligned = log_returns.squeeze().loc[index]
    vix_aligned = vix.squeeze().loc[index]
    stock_aligned = stock_close.squeeze().loc[index]

    return_disc = discretize_returns(log_returns_aligned)
    vix_disc = discretize_vix(vix_aligned)

    df = pd.DataFrame({
        'log_return': log_returns_aligned.values,
        'volatility': vix_aligned.values,
        'regime': regimes,
        'return_disc': return_disc.values,
        'volatility_disc': vix_disc.values,
        'spy_close': stock_aligned.values
    }, index=index)

    return df.dropna()