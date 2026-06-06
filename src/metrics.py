"""
Performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd

def sharpe_ratio(returns, periods=252):
    """Calculates the Sharpe Ratio given returns"""
    ann_avg = returns.mean() * periods
    ann_std_dev = returns.std() * np.sqrt(periods)
    return ann_avg/ann_std_dev

def sortino_ratio(returns, periods=252):
    """Calculates the Sortino Ratio given returns"""
    ann_avg = returns.mean() * periods
    downside = returns[returns < 0]
    downside_std_dev = downside.std() * np.sqrt(periods)
    return ann_avg/downside_std_dev

def max_drawdown(portfolio_values):
    """ """
    rolling_max = portfolio_values.cummax() 
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return drawdown.min()