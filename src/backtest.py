import math
import pandas as pd
from src.metrics import sharpe_ratio, sortino_ratio, max_drawdown, cagr

INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.001


def simulate(data, strategy, initial_capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST):
    """
    Run a backtest over data using strategy.get_action(row).
    Returns (num_trades, portfolio_values as pd.Series indexed by date).
    """
    money = initial_capital
    num_shares = 0
    num_trades = 0
    portfolio_values = []

    for row in data.itertuples():
        price = row.spy_close
        action = strategy.get_action(row)

        if action == 'buy' and money > price:
            shares_to_buy = math.floor(money / price)
            cost = price * shares_to_buy
            fee = cost * transaction_cost
            money -= cost + fee
            num_shares += shares_to_buy
            if shares_to_buy > 0:
                num_trades += 1
        elif action == 'sell' and num_shares > 0:
            proceeds = num_shares * price
            fee = proceeds * transaction_cost
            money += proceeds - fee
            num_shares = 0
            num_trades += 1

        portfolio_values.append(money + num_shares * price)

    return num_trades, pd.Series(portfolio_values, index=data.index)


def compute_metrics(portfolio_values, num_trades, initial_capital=INITIAL_CAPITAL):
    daily_returns = portfolio_values.pct_change().dropna()
    final = portfolio_values.iloc[-1]
    return {
        'final_value': final,
        'total_return_pct': (final - initial_capital) / initial_capital * 100,
        'cagr_pct': cagr(portfolio_values) * 100,
        'sharpe': sharpe_ratio(daily_returns),
        'sortino': sortino_ratio(daily_returns),
        'max_drawdown_pct': max_drawdown(portfolio_values) * 100,
        'num_trades': num_trades,
    }
