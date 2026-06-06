"""
Data loading and preprocessing for trading strategies
"""

import yfinance as yf
import pandas as pd
import numpy as np


def load_stock(ticker, start="2000-01-01"):
    """Load any stock price data from yfinance"""
    data = yf.download(ticker, start=start, auto_adjust=True)
    return data["Close"].dropna()


def load_vix(start="2000-01-01"):
    """Load VIX volatility index data"""
    vix = yf.download("^VIX", start=start, auto_adjust=True)
    return vix["Close"].dropna()


def calculate_log_returns(prices):
    """Calculate daily log returns from price series"""
    return np.log(prices / prices.shift(1)).dropna()


def load_all(ticker="SPY", start="2000-01-01"):
    """
    Load and combine stock and VIX data
    """
    stock_close = load_stock(ticker, start)
    vix_close = load_vix(start)
    log_returns = calculate_log_returns(stock_close)

    return stock_close, vix_close, log_returns