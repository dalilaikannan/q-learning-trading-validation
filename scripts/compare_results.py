"""
Rolling walk-forward comparison of Q-Learning vs Buy & Hold vs SMA.
Trains on a rolling window and reports per-window + summary metrics.
"""

import pandas as pd

from src.data import load_all
from src.regime_detection import fit_hmm, label_regimes
from src.features import build_state_space, compute_bins
from src.strategies.q_learning import QLearningStrategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.sma import SMAStrategy
from src.validation.rolling_window import RollingWindowValidator


TRAIN_CUTOFF = "2019-01-01"   # HMM fitted on data before this date
TRAIN_YEARS  = 5              # rolling training window length
TEST_MONTHS  = 12             # test window length
STEP_MONTHS  = 12             # how far to advance per roll
N_EPISODES   = 500            # Q-learning episodes per window (fewer for speed)


def _header(title, width=62):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_window_table(results, strategy_names):
    windows = results[strategy_names[0]]['windows']
    col = 14

    header = f"{'Test Period':<22}" + "".join(f"{n:>{col}}" for n in strategy_names)
    print(header)
    print("-" * len(header))

    for i, win in enumerate(windows):
        period = f"{win['test_start'].strftime('%Y-%m')} → {win['test_end'].strftime('%Y-%m')}"
        row = f"{period:<22}"
        for name in strategy_names:
            w = results[name]['windows'][i] if i < len(results[name]['windows']) else None
            val = f"{w['total_return_pct']:+.1f}%" if w else "  n/a"
            row += f"{val:>{col}}"
        print(row)


def print_summary_table(results, strategy_names):
    rows = [
        ("Mean return %",    'mean_total_return_pct', "{:+.2f}%"),
        ("Mean CAGR %",      'mean_cagr_pct',         "{:+.2f}%"),
        ("Mean Sharpe",      'mean_sharpe',            "{:.3f}"),
        ("Mean Sortino",     'mean_sortino',           "{:.3f}"),
        ("Mean max DD %",    'mean_max_drawdown_pct',  "{:.2f}%"),
        ("Win rate vs BnH",  'win_rate_vs_bnh_pct',    "{:.0f}%"),
    ]

    col = 16
    header = f"{'Metric':<24}" + "".join(f"{n:>{col}}" for n in strategy_names)
    print(header)
    print("-" * len(header))

    for label, key, fmt in rows:
        row = f"{label:<24}"
        for name in strategy_names:
            s = results[name]['summary']
            val = s.get(key)
            if val is None:
                row += f"{'—':>{col}}"
            else:
                row += f"{fmt.format(val):>{col}}"
        print(row)


if __name__ == "__main__":
    # ── 1. Load & prepare data (HMM fitted on pre-cutoff data only) ──────────
    print("Loading data...")
    stock_close, vix_close, log_returns = load_all("SPY", start="2000-01-01")

    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train_mask    = log_returns.index < cutoff
    train_returns = log_returns[train_mask]
    train_vix     = vix_close[vix_close.index < cutoff]

    print("Fitting HMM on pre-cutoff training data...")
    model, states, scaler, hmm_index = fit_hmm(train_returns, train_vix)
    regimes, label_map = label_regimes(states, train_returns.loc[hmm_index])

    train_bins = compute_bins(train_returns.loc[hmm_index], train_vix.loc[hmm_index])
    train_df = build_state_space(train_returns, train_vix, regimes,
                                  stock_close, hmm_index, bins=train_bins)

    # For rolling validation we only use the pre-cutoff period so the HMM
    # labels are always in-sample for every rolling window.
    df = train_df

    print(f"Dataset for rolling validation: {df.index[0].date()} → {df.index[-1].date()} "
          f"({len(df)} trading days)\n")

    # ── 2. Build strategies ───────────────────────────────────────────────────
    strategies = [
        ("Q-Learning",  QLearningStrategy(n_episodes=N_EPISODES)),
        ("Buy & Hold",  BuyAndHoldStrategy()),
        ("SMA (50/200)", SMAStrategy(short_window=50, long_window=200)),
    ]

    # ── 3. Run rolling window validation ─────────────────────────────────────
    validator = RollingWindowValidator(
        train_years=TRAIN_YEARS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    _header(f"ROLLING WALK-FORWARD VALIDATION  "
            f"(train={TRAIN_YEARS}y  test={TEST_MONTHS}mo  step={STEP_MONTHS}mo)")
    print()
    results = validator.run(df, strategies)

    # ── 4. Print per-window returns ───────────────────────────────────────────
    strategy_names = [name for name, _ in strategies]
    n_windows = results[strategy_names[0]]['summary']['n_windows']

    _header(f"PER-WINDOW TOTAL RETURN  ({n_windows} windows)")
    print_window_table(results, strategy_names)

    # ── 5. Print summary ──────────────────────────────────────────────────────
    _header("SUMMARY ACROSS ALL WINDOWS")
    print_summary_table(results, strategy_names)
    print()
