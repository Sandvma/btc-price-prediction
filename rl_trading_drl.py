"""
rl_trading_drl.py
===================

This module implements a baseline reinforcement‑learning agent for
cryptocurrency trading using Q‑learning.  The agent observes two
binary features derived from the daily BTC‑USD price series — the
sign of yesterday's return and whether a 3‑day moving average is
below a 7‑day moving average.  These features are combined into a
discrete state, and the agent selects among three actions (hold, buy,
sell).  Rewards are the returns realised when the position is taken
for one day.  The goal is to maximise a risk‑adjusted return, as
measured by the Sharpe ratio, though in this simplified implementation
the agent is trained with a standard Q‑learning update and evaluated
on cumulative returns and the annualised Sharpe ratio.

The script is self‑contained and does not require external RL
libraries.  It writes a summary of results to a text file and
persists the equity curve to a CSV file for further analysis.

Usage::

    python rl_trading_drl.py

The script will read the BTC‑USD price series from the configured
location, run training and evaluation, and write outputs to
``btc_price_prediction/rl_results_drl.txt`` and
``btc_price_prediction/rl_equity_curve_drl.csv``.

"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class RLConfig:
    """Configuration parameters for the Q‑learning trading agent."""

    episodes: int = 50  # Number of passes over the data for training
    epsilon: float = 0.10  # Exploration rate in epsilon‑greedy policy
    alpha: float = 0.2  # Learning rate for Q‑table updates
    gamma: float = 0.95  # Discount factor for future rewards
    window: int = 7  # Minimum history for moving averages
    data_path: str = "btc_price_prediction/data/BTC-USD.csv"  # Price data
    out_results: str = "btc_price_prediction/rl_results_drl.txt"
    out_equity: str = "btc_price_prediction/rl_equity_curve_drl.csv"


def load_prices(path: str) -> np.ndarray:
    """Load daily adjusted closing prices from a CSV file.

    Args:
        path: Path to a CSV containing the BTC‑USD data.  The file must
            include either an ``Adj Close`` column or a ``Close`` column.

    Returns:
        A numpy array of float prices.
    """
    df = pd.read_csv(path)
    if "Adj Close" in df.columns:
        close = df["Adj Close"]
    else:
        close = df["Close"]
    return close.astype(float).to_numpy()


def features_to_state(ret_sign: float, ma_short_below_long: bool) -> int:
    """Map binary features to a discrete state index.

    There are two binary features: a positive return (ret_sign > 0) and
    a boolean flag for whether the short moving average (MA3) is below
    the long moving average (MA7).  These two bits yield four
    possible states 0..3.

    Args:
        ret_sign: The daily return (not yet converted to sign).
        ma_short_below_long: Whether MA3 is below MA7.

    Returns:
        An integer representing the state.
    """
    return (1 if ret_sign > 0 else 0) + (2 if ma_short_below_long else 0)


def compute_signals(prices: np.ndarray, t: int) -> tuple[float, bool]:
    """Compute the return and moving average comparison at index ``t``.

    Args:
        prices: Array of closing prices.
        t: Current index into the price series.

    Returns:
        A tuple ``(ret, ma_short_below_long)`` where ``ret`` is the
        daily return and ``ma_short_below_long`` is True if the 3‑day
        moving average is less than the 7‑day moving average at index
        ``t``, False otherwise.
    """
    if t == 0:
        return 0.0, False
    # Compute daily return
    ret = (prices[t] / prices[t - 1]) - 1.0
    if t < 7:
        return ret, False
    ma3 = np.mean(prices[t - 3: t])
    ma7 = np.mean(prices[t - 7: t])
    return ret, (ma3 < ma7)


def train_agent(prices: np.ndarray, cfg: RLConfig) -> np.ndarray:
    """Train the Q‑learning agent.

    The Q‑table is initialised to zeros.  For each episode, the agent
    iterates through the price series, selects an action using an
    epsilon‑greedy policy, receives a reward based on the action and
    update the Q‑value according to the temporal difference rule.

    Args:
        prices: Array of prices used for training.
        cfg: Configuration with hyperparameters.

    Returns:
        The learned Q‑table as a 4x3 numpy array.
    """
    n_states = 4
    n_actions = 3
    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(123)  # Deterministic exploration
    for _ in range(cfg.episodes):
        pos = 0
        for t in range(1, len(prices)):
            ret, flag = compute_signals(prices, t)
            state = features_to_state(ret, flag)
            # Epsilon‑greedy action selection
            if rng.random() < cfg.epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q[state]))
            # Compute reward using the next return and the action taken
            if t < len(prices) - 1:
                next_ret, next_flag = compute_signals(prices, t + 1)
            else:
                next_ret, next_flag = (0.0, False)
            next_state = features_to_state(next_ret, next_flag)
            next_pos = {0: 0, 1: +1, 2: -1}[action]
            reward = next_pos * next_ret
            # Q-learning update
            Q[state, action] = (
                (1 - cfg.alpha) * Q[state, action]
                + cfg.alpha * (reward + cfg.gamma * np.max(Q[next_state]))
            )
            pos = next_pos
    return Q


def derive_policy(Q: np.ndarray) -> List[int]:
    """Compute the deterministic policy from a Q‑table.

    Args:
        Q: Learned Q‑values array of shape (4, 3).

    Returns:
        List of actions (0 hold, 1 buy, 2 sell) for each state.
    """
    return [int(np.argmax(Q[s])) for s in range(Q.shape[0])]


def evaluate_policy(prices: np.ndarray, policy: List[int]) -> tuple[np.ndarray, float, float]:
    """Evaluate a policy on the price data.

    The evaluation assumes trades are closed at the end of each day.
    Returns the equity curve, the cumulative return and the annualised
    Sharpe ratio.

    Args:
        prices: Array of closing prices.
        policy: List mapping each state to an action.

    Returns:
        Tuple ``(equity_curve, cum_return, sharpe)`` where
        ``equity_curve`` is an array of equity values, ``cum_return``
        is the total return and ``sharpe`` is the annualised Sharpe ratio.
    """
    pos = 0
    equity = [1.0]
    for t in range(1, len(prices)):
        ret, flag = compute_signals(prices, t)
        state = features_to_state(ret, flag)
        # apply reward from previous position
        equity[-1] = equity[-1] * (1.0 + pos * ret)
        action = policy[state]
        pos = {0: 0, 1: +1, 2: -1}[action]
        equity.append(equity[-1])
    eq_arr = np.array(equity)
    # Use log returns for Sharpe ratio calculation
    log_returns = np.diff(np.log(eq_arr.clip(min=1e-12)))
    sharpe = float(
        np.sqrt(252) * (log_returns.mean() / (log_returns.std() + 1e-12))
    )
    cum_return = float(eq_arr[-1]) - 1.0
    return eq_arr, cum_return, sharpe


def run(cfg: RLConfig) -> str:
    """Run the entire training and evaluation pipeline.

    This function orchestrates loading the data, training the Q‑learning
    agent, deriving a policy, evaluating it, persisting the equity
    curve and summary, and returning the summary string.

    Args:
        cfg: Configuration for file paths and hyperparameters.

    Returns:
        A formatted summary string of the results.
    """
    prices = load_prices(cfg.data_path)
    Q = train_agent(prices, cfg)
    policy = derive_policy(Q)
    equity_curve, cum_return, sharpe = evaluate_policy(prices, policy)
    # Persist outputs
    eq_df = pd.DataFrame({"step": np.arange(len(equity_curve)), "equity": equity_curve})
    eq_df.to_csv(cfg.out_equity, index=False)
    summary = (
        "RL Q-learning (estado= sign(ret) x (MA3<MA7), acciones=hold/buy/sell)\n"
        f"Episodios: {cfg.episodes}, epsilon: {cfg.epsilon}, alpha: {cfg.alpha}, gamma: {cfg.gamma}\n"
        f"Retorno acumulado: {1 + cum_return:.4f}\n"
        f"Sharpe anualizado: {sharpe:.4f}\n"
        f"Política (estado→acción): {policy}\n"
    )
    with open(cfg.out_results, "w") as f:
        f.write(summary)
    return summary


def main() -> None:
    cfg = RLConfig()
    summary = run(cfg)
    print(summary)


if __name__ == "__main__":
    main()