"""
Market regime detection using Hidden Markov Models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


def fit_hmm(log_returns, vix, n_states=3, random_state=42):
    """
    Fit Gaussian HMM to detect market regimes

    Args:
        log_returns: Series of log returns
        vix: Series of VIX values
        n_states: Number of hidden states (default 3)
        random_state: Random seed for reproducibility

    Returns:
        model: Fitted HMM model
        states: Array of predicted states
        scaler: Fitted scaler 
    """
    df = pd.concat([log_returns, vix], axis=1).dropna()

    # scale features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # fit hmm
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=random_state
    )
    model.fit(df_scaled)

    # predict states
    states = model.predict(df_scaled)

    return model, states, scaler, df.index


def label_regimes(states, log_returns):
    """
    Label HMM states as bull/bear/sideways based on mean returns
    """
    state_means = {}
    for state in np.unique(states):
        mask = states == state
        state_means[state] = float(log_returns.values[mask].mean())

    sorted_states = sorted(state_means.items(), key=lambda x: x[1], reverse=True)

    label_map = {
        sorted_states[0][0]: "bull",
        sorted_states[1][0]: "sideways",
        sorted_states[2][0]: "bear"
    }

    return pd.Categorical([label_map[s] for s in states],
                          categories=["bull", "sideways", "bear"])