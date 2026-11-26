import pandas as pd

from src.data import load_all
from src.regime_detection import fit_hmm, label_regimes, predict_regimes
from src.features import build_state_space, compute_bins
from src.backtest import simulate, compute_metrics, INITIAL_CAPITAL
from src.strategies.q_learning import QLearningStrategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.sma import SMAStrategy


TRAIN_END = "2019-01-01"


def print_results(name, metrics):
    print(f"\n=== {name} ===")
    print(f"  Final value  : ${metrics['final_value']:,.2f}")
    print(f"  Total return : {metrics['total_return_pct']:.2f}%")
    print(f"  CAGR         : {metrics['cagr_pct']:.2f}%")
    print(f"  Sharpe       : {metrics['sharpe']:.3f}")
    print(f"  Sortino      : {metrics['sortino']:.3f}")
    print(f"  Max drawdown : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Trades       : {metrics['num_trades']}")


if __name__ == "__main__":
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    stock_close, vix_close, log_returns = load_all("SPY", start="2000-01-01")

    train_end = pd.Timestamp(TRAIN_END)

    # ── 2. Split raw data before any fitting (no data leakage) ───────────────
    train_mask = log_returns.index < train_end
    train_returns = log_returns[train_mask]
    train_vix     = vix_close[vix_close.index < train_end]
    test_returns  = log_returns[~train_mask]
    test_vix      = vix_close[vix_close.index >= train_end]

    # ── 3. Fit HMM on train data only ─────────────────────────────────────────
    print("Fitting HMM on training data...")
    model, train_states, scaler, train_index = fit_hmm(train_returns, train_vix)
    train_regimes, label_map = label_regimes(train_states, train_returns.loc[train_index])

    # ── 4. Apply fitted HMM to test data (no future information used) ─────────
    test_regimes, test_index = predict_regimes(model, scaler, label_map, test_returns, test_vix)

    # ── 5. Discretize features using train-period bin edges ───────────────────
    train_bins = compute_bins(train_returns.loc[train_index], train_vix.loc[train_index])

    train_df = build_state_space(train_returns, train_vix, train_regimes,
                                  stock_close, train_index, bins=train_bins)
    test_df  = build_state_space(test_returns,  test_vix,  test_regimes,
                                  stock_close, test_index,  bins=train_bins)

    print(f"Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} days)")
    print(f"Test : {test_df.index[0].date()}  → {test_df.index[-1].date()}  ({len(test_df)} days)\n")

    # ── 6. Train Q-learning on train set ─────────────────────────────────────
    print("Training Q-learning agent...")
    agent = QLearningStrategy(n_episodes=1000)
    agent.train(train_df)

    # ── 7. Train SMA on combined data so test-period signals are available ────
    combined_df = pd.concat([train_df, test_df])
    sma = SMAStrategy(short_window=50, long_window=200)
    sma.train(combined_df)

    bnh = BuyAndHoldStrategy()

    # ── 8. Backtest all strategies on test set ────────────────────────────────
    ql_trades,  ql_values  = simulate(test_df, agent)
    bnh_trades, bnh_values = simulate(test_df, bnh)
    sma_trades, sma_values = simulate(test_df, sma)

    ql_metrics  = compute_metrics(ql_values,  ql_trades)
    bnh_metrics = compute_metrics(bnh_values, bnh_trades)
    sma_metrics = compute_metrics(sma_values, sma_trades)

    # ── 9. Print comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"RESULTS  (test period: {TRAIN_END} – present)")
    print("=" * 50)

    print_results("Q-Learning Agent", ql_metrics)
    print_results("Buy & Hold",       bnh_metrics)
    print_results("SMA (50/200)",     sma_metrics)

    print("\n" + "=" * 50)
    print("COMPARISON vs BUY & HOLD")
    print("=" * 50)
    bnh_final = bnh_metrics['final_value']
    for name, m in [("Q-Learning", ql_metrics), ("SMA (50/200)", sma_metrics)]:
        diff = ((m['final_value'] / bnh_final) - 1) * 100
        symbol = "+" if m['final_value'] > bnh_final else "-"
        label  = "outperformed" if m['final_value'] > bnh_final else "underperformed"
        print(f"  [{symbol}] {name} {label} by {abs(diff):.2f}%")
