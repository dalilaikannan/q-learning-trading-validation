"""
Feature engineering for Q-learning state space & converts raw market data into discrete state representations
"""

import numpy as np
import pandas as pd


def discretize_returns(log_returns):
    log_returns = log_returns.squeeze()
    low = log_returns.quantile(0.33)
    high = log_returns.quantile(0.67)
    return pd.cut(log_returns, bins=[-np.inf, low, high, np.inf], labels=["low", "medium", "high"])


def discretize_vix(vix):
    vix = vix.squeeze()
    low = vix.quantile(0.33)
    high = vix.quantile(0.67)
    return pd.cut(vix, bins=[-np.inf, low, high, np.inf], labels=["low", "neutral", "high"])


def compute_bins(log_returns, vix):
    """Compute discretization bin edges from a training period for reuse on test data."""
    log_returns = log_returns.squeeze()
    vix = vix.squeeze()
    return {
        'return_low': float(log_returns.quantile(0.33)),
        'return_high': float(log_returns.quantile(0.67)),
        'vix_low': float(vix.quantile(0.33)),
        'vix_high': float(vix.quantile(0.67)),
    }


def apply_bins(log_returns, vix, bins):
    """Discretize returns and VIX using precomputed bin edges."""
    log_returns = log_returns.squeeze()
    vix = vix.squeeze()
    return_disc = pd.cut(
        log_returns,
        bins=[-np.inf, bins['return_low'], bins['return_high'], np.inf],
        labels=["low", "medium", "high"]
    )
    vix_disc = pd.cut(
        vix,
        bins=[-np.inf, bins['vix_low'], bins['vix_high'], np.inf],
        labels=["low", "neutral", "high"]
    )
    return return_disc, vix_disc


def build_state_space(log_returns, vix, regimes, stock_close, index, bins=None):
    """
    Build the full state space DataFrame for Q-learning.
    Pass bins (from compute_bins) to apply training-period discretization to new data.
    """
    log_returns_aligned = log_returns.squeeze().loc[index]
    vix_aligned = vix.squeeze().loc[index]
    stock_aligned = stock_close.squeeze().loc[index]

    if bins is None:
        return_disc = discretize_returns(log_returns_aligned)
        vix_disc = discretize_vix(vix_aligned)
    else:
        return_disc, vix_disc = apply_bins(log_returns_aligned, vix_aligned, bins)

    df = pd.DataFrame({
        'log_return': log_returns_aligned.values,
        'volatility': vix_aligned.values,
        'regime': regimes,
        'return_disc': return_disc.values,
        'volatility_disc': vix_disc.values,
        'spy_close': stock_aligned.values
    }, index=index)

    return df.dropna()