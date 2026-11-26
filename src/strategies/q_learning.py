from src.strategies.base import Strategy
import numpy as np
import random
import math


class QLearningStrategy(Strategy):
    """Q-learning trading agent with regime-conditioned state space"""

    REGIMES = ["bull", "sideways", "bear"]
    RETURN_BINS = ["low", "medium", "high"]
    VIX_BINS = ["low", "neutral", "high"]
    ACTIONS = ['buy', 'sell', 'hold']

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01,
                 transaction_cost=0.001, n_episodes=1000):
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.transaction_cost = transaction_cost
        self.n_episodes = n_episodes
        self.reset()

    def reset(self):
        self.epsilon = self._epsilon_start
        self.q_table = {}
        for r in self.REGIMES:
            for rb in self.RETURN_BINS:
                for vb in self.VIX_BINS:
                    self.q_table[(r, rb, vb)] = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}

    def train(self, data):
        """
        Train on a DataFrame with columns:
        regime, return_disc, volatility_disc, spy_close
        """
        episode_rewards = []
        episode_trades = []

        for episode in range(self.n_episodes):
            money = 10000
            num_shares = 0
            total_reward = 0
            num_trades = 0

            for i, row in enumerate(data.itertuples(index=False)):
                current_state = (str(row.regime), str(row.return_disc), str(row.volatility_disc))
                current_price = row.spy_close
                before_value = (num_shares * current_price) + money

                if random.random() < self.epsilon:
                    action = random.choice(self.ACTIONS)
                else:
                    action = max(self.q_table[current_state],
                                 key=lambda a: self.q_table[current_state][a])

                if action == 'buy' and money > current_price:
                    shares_to_buy = math.floor(money / current_price)
                    cost = current_price * shares_to_buy
                    fee = cost * self.transaction_cost
                    money -= cost + fee
                    num_shares += shares_to_buy
                    if shares_to_buy > 0:
                        num_trades += 1

                elif action == 'sell' and num_shares > 0:
                    proceeds = num_shares * current_price
                    fee = proceeds * self.transaction_cost
                    money += proceeds - fee
                    num_shares = 0
                    num_trades += 1

                if i + 1 < len(data):
                    next_row = data.iloc[i + 1]
                    next_price = next_row.spy_close
                    after_value = (num_shares * next_price) + money

                    portfolio_return = (after_value - before_value) / before_value
                    market_return = np.exp(next_row.log_return) - 1
                    reward = portfolio_return - market_return
                    total_reward += reward

                    next_state = (str(next_row.regime), str(next_row.return_disc), str(next_row.volatility_disc))

                    current_q = self.q_table[current_state][action]
                    max_next_q = max(self.q_table[next_state].values())
                    self.q_table[current_state][action] = current_q + self.alpha * (
                        reward + self.gamma * max_next_q - current_q
                    )

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)
            episode_trades.append(num_trades)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}/{self.n_episodes} | "
                      f"Avg Reward (last 50): {avg_reward:.4f} | "
                      f"Epsilon: {self.epsilon:.3f} | Trades: {num_trades}")

        print("\nTraining complete!")
        print(f"Final epsilon: {self.epsilon:.3f}")
        print(f"Average reward: {np.mean(episode_rewards):.4f}")

    def predict(self, state):
        """Return best action for state tuple (regime, return_disc, volatility_disc)."""
        state = tuple(str(s) for s in state)
        return max(self.q_table[state], key=lambda a: self.q_table[state][a])

    def get_action(self, row) -> str:
        return self.predict((str(row.regime), str(row.return_disc), str(row.volatility_disc)))
