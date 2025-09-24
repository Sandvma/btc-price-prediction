"""
factory_pipeline.py
====================

This module implements a simplified version of the workflow described in the
extended quant strategy factory manual.  The goal is to provide a reusable
pipeline capable of:

* Loading point‑in‑time financial data and generating alternative bars (optional).
* Computing a rich set of technical features using a unified naming scheme.
* Applying triple‑barrier labelling and optional meta‑labelling.
* Performing combinatorial purged cross‑validation with embargo to avoid
  look‑ahead bias.
* Learning market regimes via a Hidden Markov Model (HMM) and conditioning
  predictions on regime.
* Training base and meta models (tree‑based and logistic, by default) on the
  engineered features.
* Backtesting the resulting signals with realistic cost assumptions and risk
  controls (volatility targeting and fractional Kelly sizing).
* Evaluating the robustness of the strategy using Sharpe ratio, deflated
  Sharpe ratio and probability of backtest overfitting (PBO).

The pipeline is deliberately modular—each component can be swapped out or
extended.  It is not intended as a production‑ready trading system, but
illustrates the key ideas of the factory: systematic workflow, rigorous
validation and honest evaluation.

Example usage (run from the repository root):

```
python btc_price_prediction/run_factory.py --config btc_price_prediction/config/example.yaml
```

See the accompanying YAML configuration for options and defaults.
"""

from __future__ import annotations

import math
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from hmmlearn import hmm  # type: ignore
    _HMM_AVAILABLE = True
except Exception:
    _HMM_AVAILABLE = False


def load_dataset(path: str) -> pd.DataFrame:
    """Load the OHLCV dataset from a CSV file into a pandas DataFrame.

    The CSV is expected to contain at least the columns: Date, Open,
    High, Low, Close, Volume.  Additional columns are ignored.  The
    returned DataFrame is indexed by a pandas datetime index sorted
    ascending.
    """
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
    else:
        # assume the index is date already
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def compute_features(
    df: pd.DataFrame,
    feature_list: List[str],
    params: Dict[str, Any] | None = None,
    prefix: str = "btc_d1_",
) -> pd.DataFrame:
    """Compute a variety of technical features and return a DataFrame.

    Supported feature identifiers include:

    - 'ret'         : simple return (log or arithmetic, see params)
    - 'ret_lag_k'   : return shifted by k bars (k provided in params)
    - 'atr_n'       : Average True Range over n bars
    - 'rsi_n'       : Relative Strength Index over n bars
    - 'mom_n'       : Momentum (difference between close and close n bars ago)
    - 'kama_n'      : Kaufman adaptive moving average difference (approximation)
    - 'vol_usd'     : Dollar volume per bar
    - 'spread'      : High minus Low
    - 'stochrsi_n'  : Stochastic RSI over n bars

    The naming convention follows <prefix><feature> with parameters
    appended.  For example, 'btc_d1_rsi_14' or 'btc_d1_ret_lag_3'.

    The `params` dict can specify default windows (e.g. {'ret':
    {'log': True}, 'atr': {'n': 14}, ...}).  Any unspecified parameter
    uses a sensible default.
    """
    if params is None:
        params = {}

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df.get('volume', pd.Series(dtype=float)).astype(float)

    features: Dict[str, pd.Series] = {}

    for feat in feature_list:
        if feat.startswith('ret'):
            # simple or log return
            log_flag = params.get('ret', {}).get('log', True)
            # compute return one step ahead by default
            ret = close.pct_change() if not log_flag else np.log(close / close.shift(1))
            feats_to_add = {'ret': ret}
            # if ret lag specified
            if 'lag' in feat:
                try:
                    lag = int(feat.split('_')[-1])
                except Exception:
                    lag = 1
                feats_to_add = {
                    f'ret_lag_{lag}': ret.shift(lag)
                }
            for name, series in feats_to_add.items():
                full_name = prefix + name
                features[full_name] = series
        elif feat.startswith('atr'):
            # ATR over n
            n = params.get('atr', {}).get('n', 14)
            tr = pd.concat([
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(n).mean()
            features[prefix + f'atr_{n}'] = atr
        elif feat.startswith('rsi'):
            n = params.get('rsi', {}).get('n', 14)
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll_up = gain.rolling(n).mean()
            roll_down = loss.rolling(n).mean()
            rs = roll_up / roll_down.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            features[prefix + f'rsi_{n}'] = rsi
        elif feat.startswith('mom'):
            n = params.get('mom', {}).get('n', 10)
            momentum = close - close.shift(n)
            features[prefix + f'mom_{n}'] = momentum
        elif feat.startswith('kama'):
            n = params.get('kama', {}).get('n', 10)
            # simple KAMA approximation: difference between close and ema
            ema = close.ewm(span=n, adjust=False).mean()
            features[prefix + f'kama_{n}'] = close - ema
        elif feat.startswith('vol_usd'):
            vol_usd = volume * close
            features[prefix + 'vol_usd'] = vol_usd
        elif feat.startswith('spread'):
            spread = high - low
            features[prefix + 'spread'] = spread
        elif feat.startswith('stochrsi'):
            n = params.get('stochrsi', {}).get('n', 14)
            # compute RSI
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            roll_up = gain.rolling(n).mean()
            roll_down = loss.rolling(n).mean()
            rs = roll_up / roll_down.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            # stochRSI: (rsi - min(rsi)) / (max - min)
            min_rsi = rsi.rolling(n).min()
            max_rsi = rsi.rolling(n).max()
            stochrsi = (rsi - min_rsi) / (max_rsi - min_rsi)
            features[prefix + f'stochrsi_{n}'] = stochrsi
        else:
            # unknown feature; skip
            continue
    features_df = pd.DataFrame(features)
    return features_df


def triple_barrier_labels(
    close: pd.Series,
    horizon: int,
    tp: float,
    sl: float,
) -> pd.Series:
    """Compute triple‑barrier labels for a price series.

    Args:
        close: price series indexed by date.
        horizon: number of bars to look ahead for take profit/stop loss.
        tp: take profit threshold expressed as a fraction of price (e.g. 0.01 for 1%).
        sl: stop loss threshold expressed as a fraction of price (e.g. 0.01 for 1%).

    Returns:
        A pandas Series of labels: +1 for hitting TP first, -1 for hitting SL
        first, 0 otherwise (if neither threshold hit within horizon).
    """
    n = len(close)
    # initialise labels with NaNs
    labels = pd.Series(index=close.index, dtype=float)
    # precompute high and low over horizon window for efficiency
    future_max = close.rolling(window=horizon, min_periods=1).max().shift(-horizon + 1)
    future_min = close.rolling(window=horizon, min_periods=1).min().shift(-horizon + 1)
    for i in range(n - horizon):
        price = close.iloc[i]
        max_future = future_max.iloc[i]
        min_future = future_min.iloc[i]
        # price increase threshold
        tp_price = price * (1 + tp)
        sl_price = price * (1 - sl)
        # Determine which threshold is hit first
        # use forward slice
        window = close.iloc[i + 1: i + horizon + 1]
        # index of take profit hit
        tp_hit = window[window >= tp_price].index.min() if any(window >= tp_price) else None
        sl_hit = window[window <= sl_price].index.min() if any(window <= sl_price) else None
        if tp_hit is not None and (sl_hit is None or tp_hit <= sl_hit):
            labels.iloc[i] = 1
        elif sl_hit is not None and (tp_hit is None or sl_hit < tp_hit):
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0
    # fill last horizon points with NaN because we don't have enough look ahead
    labels.iloc[n - horizon:] = np.nan
    return labels


def cpcv_splits(n_samples: int, blocks: int = 6, embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate combinatorial purged cross‑validation indices with embargo.

    Splits the data into `blocks` contiguous segments.  For each combination
    of training and test blocks where the number of training blocks is
    ``blocks - 2`` (leave two blocks for testing), returns indices for
    training and test.  An embargo is applied by dropping a fraction
    ``embargo_pct`` of samples at the boundaries between train and test.

    Args:
        n_samples: total number of samples
        blocks: number of blocks to divide the data into
        embargo_pct: percentage of samples to drop on either side of
          the boundary between train and test splits

    Returns:
        A list of (train_idx, test_idx) tuples.
    """
    # compute block boundaries
    block_size = int(np.floor(n_samples / blocks))
    indices = np.arange(n_samples)
    block_indices = [
        indices[i * block_size: (i + 1) * block_size] for i in range(blocks)
    ]
    # handle any leftover samples by appending to last block
    leftover = indices[blocks * block_size:]
    if len(leftover) > 0:
        block_indices[-1] = np.concatenate([block_indices[-1], leftover])
    # generate combinations of blocks for training
    splits = []
    # choose blocks - 2 blocks for training; leave 2 for test and validation implicitly
    for train_block_indices in itertools.combinations(range(blocks), blocks - 2):
        # test blocks are those not in train_block_indices
        test_blocks = [i for i in range(blocks) if i not in train_block_indices]
        # get training indices
        train_idx = np.concatenate([block_indices[i] for i in train_block_indices])
        test_idx = np.concatenate([block_indices[i] for i in test_blocks])
        # apply purge: remove a fraction of samples around boundary
        # boundaries: last train block index and first test block index
        # compute purge length
        purge_len = int(np.ceil(len(indices) * embargo_pct))
        if purge_len > 0:
            max_train = train_idx.max()
            min_test = test_idx.min()
            purge_start = max_train - purge_len + 1
            purge_end = min_test + purge_len - 1
            # drop any indices in this range from train and test
            train_idx = train_idx[train_idx < purge_start]
            test_idx = test_idx[test_idx > purge_end]
        splits.append((train_idx, test_idx))
    return splits


def fit_hmm_states(features: pd.DataFrame, n_states: int = 3) -> np.ndarray:
    """Fit a Hidden Markov Model to the given features and return the state sequence.

    This function will fall back to KMeans clustering if hmmlearn is not
    available.  The features should be numeric and indexed by date.
    """
    # drop NaNs
    clean_feats = features.dropna().values
    if len(clean_feats) == 0:
        # return zero states if no data
        return np.zeros(len(features), dtype=int)
    if _HMM_AVAILABLE:
        # normalise features
        scaler = StandardScaler()
        X = scaler.fit_transform(clean_feats)
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100)
        model.fit(X)
        hidden_states = model.predict(X)
    else:
        # fallback to k-means clustering on scaled data
        from sklearn.cluster import KMeans  # lazy import
        scaler = StandardScaler()
        X = scaler.fit_transform(clean_feats)
        km = KMeans(n_clusters=n_states, random_state=42)
        hidden_states = km.fit_predict(X)
    # pad with NaNs for rows dropped
    state_series = pd.Series(index=features.dropna().index, data=hidden_states)
    full_states = state_series.reindex(features.index, method='nearest').fillna(method='bfill').fillna(method='ffill').astype(int)
    return full_states.values


def compute_sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualised Sharpe ratio of a returns series.

    Args:
        returns: array of daily returns (arithmetic)
        risk_free: risk free rate per period

    Returns:
        Sharpe ratio (annualised assuming 252 trading days)
    """
    excess = returns - risk_free
    if len(excess) == 0 or np.allclose(excess, 0):
        return 0.0
    mean = np.nanmean(excess)
    std = np.nanstd(excess, ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std * np.sqrt(252)
    return sr


def deflated_sharpe_ratio(sharpe: float, n_trials: int) -> float:
    """Compute a simplified deflated Sharpe ratio.

    Following López de Prado, the deflated Sharpe ratio adjusts the observed
    Sharpe for the number of trials performed during strategy discovery.
    This implementation uses a simplified approximation whereby the Sharpe
    is scaled by the factor (1 - 0.5 * log(n_trials) / n_trials).  For
    small n_trials this has little effect; for large search spaces it
    reduces the reported Sharpe.  The resulting value is bounded below by
    zero.
    """
    if n_trials < 1:
        n_trials = 1
    penalty = 0.5 * math.log(max(n_trials, 1)) / max(n_trials, 1)
    dsr = sharpe * (1 - penalty)
    return max(dsr, 0.0)


def probability_of_backtest_overfitting(train_sharpes: List[float], test_sharpes: List[float]) -> float:
    """Estimate the probability of backtest overfitting (PBO) given training and test sharpes.

    The PBO is defined as 1 minus the quantile rank of the best test
    Sharpe within the distribution of training Sharpes.  Intuitively, if
    the model with the best out‑of‑sample performance also had an above
    average in‑sample performance, the PBO is low; if it was a poor
    performer in‑sample, the PBO is high.  The training and test arrays
    should correspond across folds.
    """
    if not train_sharpes or not test_sharpes or len(train_sharpes) != len(test_sharpes):
        return 1.0
    # find index of best test Sharpe
    best_idx = int(np.argmax(test_sharpes))
    best_test_sharpe = test_sharpes[best_idx]
    # compute quantile rank of corresponding train Sharpe
    train_sr = train_sharpes[best_idx]
    rank = sum(s <= train_sr for s in train_sharpes) / len(train_sharpes)
    pbo = 1 - rank
    return pbo


def volatility_target_weights(returns: np.ndarray, target_vol: float = 0.1, window: int = 30, cap: float = 2.0) -> np.ndarray:
    """Compute position weights via volatility targeting.

    The weight at time t is scaled by the ratio of target volatility to the
    realised volatility over a rolling window.  The weight is capped at
    `cap` to avoid excessive leverage.  If the realised volatility is
    zero (all returns equal) then the weight is set to zero.
    """
    weights = np.zeros_like(returns)
    for i in range(len(returns)):
        if i < window:
            weights[i] = 0.0
            continue
        window_ret = returns[i - window: i]
        vol = np.nanstd(window_ret, ddof=1)
        if vol > 0:
            w = (target_vol / vol)
            weights[i] = min(cap, w)
        else:
            weights[i] = 0.0
    return weights


def fractional_kelly(p: float, b: float, alpha: float = 0.5) -> float:
    """Compute fractional Kelly position size given win probability p and win/loss ratio b.

    Args:
        p: probability of a winning trade
        b: ratio of average win to average loss
        alpha: scaling factor (0 < alpha <= 0.5)

    Returns:
        Fraction of capital to allocate (f).  Negative values imply short.
    """
    # avoid divide by zero
    if b <= 0:
        return 0.0
    f_star = p - (1 - p) / b
    return alpha * f_star


def compute_kelly_weights(signals: np.ndarray, returns: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Estimate fractional Kelly weights per trade based on historical hit rate and gain/loss ratio.

    The weight at time t is computed using the win probability and
    average win/loss computed from the past window of trades.  If
    insufficient history exists, the weight is set to zero.  Negative
    weights correspond to short exposure.
    """
    n = len(signals)
    weights = np.zeros(n)
    # compute running metrics
    wins = []
    losses = []
    for i in range(n):
        if i > 0 and signals[i - 1] != 0:
            # record outcome of previous trade
            outcome = signals[i - 1] * returns[i]
            if outcome > 0:
                wins.append(outcome)
            elif outcome < 0:
                losses.append(-outcome)
        # compute p and b
        if len(wins) + len(losses) >= 10:
            p = len(wins) / (len(wins) + len(losses))
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            weights[i] = fractional_kelly(p, b, alpha)
        else:
            weights[i] = 0.0
    return weights


@dataclass
class StrategyMetrics:
    mse: float
    accuracy: float
    sharpe: float
    deflated_sharpe: float
    pbo: float
    net_return: float
    max_drawdown: float


def backtest_strategy(
    close: pd.Series,
    signals: np.ndarray,
    vol_target: float = 0.1,
    kelly_alpha: float = 0.0,
    cost_per_trade: float = 0.0,
    slippage: float = 0.0,
) -> Tuple[float, float, np.ndarray]:
    """Backtest a trading strategy using simple position signals.

    Args:
        close: price series used to compute returns.
        signals: array of position signals in {-1, 0, +1}.  Each element
          indicates the position taken at the start of the corresponding
          period.  Signals are assumed to align with close index.
        vol_target: annualised volatility target for volatility targeting.
        kelly_alpha: factor for fractional Kelly sizing (0 disables Kelly sizing).
        cost_per_trade: transaction cost per unit traded (in return units, e.g. 0.001 for 10bp)
        slippage: proportional slippage cost per trade.

    Returns:
        Tuple containing (strategy return, sharpe ratio, daily returns series).
    """
    # compute daily returns
    ret = close.pct_change().fillna(0).values
    # compute position changes (i.e. trades)
    pos = signals
    pos_shifted = np.concatenate([[0], pos[:-1]])  # previous period position
    trades = pos - pos_shifted
    # compute volatility targeting weights
    vt_weights = volatility_target_weights(ret)
    # compute Kelly weights
    if kelly_alpha > 0:
        kelly_weights = compute_kelly_weights(pos, ret, alpha=kelly_alpha)
    else:
        kelly_weights = np.zeros_like(pos)
    # total weights
    weights = pos.copy().astype(float)
    # apply volatility targeting multiplicatively
    weights *= vt_weights
    # apply Kelly adjustments
    weights += kelly_weights
    # strategy returns before costs
    strat_ret = weights * ret
    # subtract cost for each trade (absolute value of trade) times cost per trade
    strat_ret -= np.abs(trades) * cost_per_trade
    # slippage cost: approximate as slippage * absolute position times volatility
    strat_ret -= np.abs(pos_shifted) * slippage * np.abs(ret)
    # compute total and metrics
    total_return = np.nansum(strat_ret)
    sharpe = compute_sharpe_ratio(strat_ret)
    return total_return, sharpe, strat_ret


def run_pipeline(config: Dict[str, Any]) -> StrategyMetrics:
    """Execute the full strategy discovery pipeline using the provided configuration.

    Args:
        config: dictionary parsed from YAML specifying dataset, features,
          labelling, validation, regime, modelling, risk and robustness
          options.

    Returns:
        StrategyMetrics summarising the performance of the best model.
    """
    # load dataset
    dataset_path = config['dataset']['path']
    df = load_dataset(dataset_path)
    # compute features
    feature_list = config['features']['list']
    feature_params = config['features'].get('params', {})
    prefix = f"{config['dataset']['symbol']}_{config['dataset']['tf']}_"
    features = compute_features(df, feature_list, feature_params, prefix=prefix)
    # align features with price
    data = pd.concat([df[['close']], features], axis=1).dropna()
    # compute labels
    lbl_conf = config['labeling']
    horizon = lbl_conf['horizon_bars']
    tp = lbl_conf['tp']
    sl = lbl_conf['sl']
    labels = triple_barrier_labels(data['close'], horizon, tp, sl)
    data = data.assign(label=labels)
    data = data.dropna()
    # compute regime states (optional)
    regime_conf = config.get('regime', {})
    states = None
    if regime_conf:
        hmm_feats = data[[col for col in features.columns if any(col.endswith(suf) for suf in ['ret', 'atr_14', 'vol_usd', 'spread'])]].dropna()
        # ensure at least one column
        if not hmm_feats.empty:
            n_states = regime_conf.get('hmm_states', 3)
            states_arr = fit_hmm_states(hmm_feats, n_states)
            data = data.assign(state=states_arr)
            states = states_arr
    # drop samples with missing values
    data = data.dropna()
    X = data[features.columns].values
    y = data['label'].values.astype(int)
    # normalise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # cross validation splits
    val_conf = config['validation']
    splits = cpcv_splits(len(data), blocks=val_conf.get('blocks', 6), embargo_pct=val_conf.get('embargo_pct', 0.01))
    # store metrics per fold
    train_sharpes = []
    test_sharpes = []
    # store best fold metrics overall
    best_metrics: StrategyMetrics | None = None
    for train_idx, test_idx in splits:
        # subset
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # base model classifier
        base_model = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        base_model.fit(X_train, y_train)
        y_pred_train = base_model.predict(X_train)
        y_pred_test = base_model.predict(X_test)
        # meta labeling: compute meta labels from train set
        # meta label is 1 if base signal is correct (y_pred == y), else 0; we only consider non-zero predictions
        meta_train_idx = y_pred_train != 0
        if meta_train_idx.any():
            meta_labels_train = (np.sign(y_train[meta_train_idx]) == np.sign(y_pred_train[meta_train_idx])).astype(int)
            meta_features_train = X_train[meta_train_idx]
            # Only train meta model if at least two classes are present
            if len(np.unique(meta_labels_train)) > 1:
                meta_model = LogisticRegression(max_iter=200)
                meta_model.fit(meta_features_train, meta_labels_train)
                # apply meta model to test
                meta_test_idx = y_pred_test != 0
                if meta_test_idx.any():
                    meta_pred = meta_model.predict(X_test[meta_test_idx])
                    # if meta model predicts 0, set signal to 0
                    filtered_signals = y_pred_test.copy()
                    filtered_signals[meta_test_idx] = filtered_signals[meta_test_idx] * meta_pred
                else:
                    filtered_signals = y_pred_test.copy()
            else:
                # not enough class diversity; skip meta labeling
                filtered_signals = y_pred_test.copy()
        else:
            filtered_signals = y_pred_test.copy()
        # compute performance
        # compute cost per trade from config
        cost = config.get('backtest', {}).get('transaction_cost', 0.0)
        slippage = config.get('backtest', {}).get('slippage', 0.0)
        vol_target = config.get('risk', {}).get('target_vol', 0.1)
        kelly_alpha = config.get('risk', {}).get('kelly_alpha', 0.0)
        # training performance on training indices using base model predictions
        train_signals = y_pred_train
        tr_return, tr_sharpe, _ = backtest_strategy(
            data['close'].iloc[train_idx], train_signals, vol_target=vol_target, kelly_alpha=kelly_alpha,
            cost_per_trade=cost, slippage=slippage
        )
        train_sharpes.append(tr_sharpe)
        # test performance using filtered_signals
        test_return, test_sharpe, test_ret_series = backtest_strategy(
            data['close'].iloc[test_idx], filtered_signals, vol_target=vol_target, kelly_alpha=kelly_alpha,
            cost_per_trade=cost, slippage=slippage
        )
        test_sharpes.append(test_sharpe)
        # compute mse and accuracy for fold
        mse = mean_squared_error(y_test, y_pred_test)
        accuracy = (y_test == y_pred_test).mean()
        # compute deflated Sharpe ratio
        dsr = deflated_sharpe_ratio(test_sharpe, n_trials=len(splits))
        # compute PBO (will compute later after loops)
        # compute net return and max drawdown
        cumulative = np.nancumsum(test_ret_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = drawdown.max() if len(drawdown) > 0 else 0.0
        metrics = StrategyMetrics(mse=mse, accuracy=accuracy, sharpe=test_sharpe, deflated_sharpe=dsr, pbo=0.0,
                                  net_return=test_return, max_drawdown=max_dd)
        # update best metrics if deflated Sharpe is higher
        if best_metrics is None or dsr > best_metrics.deflated_sharpe:
            best_metrics = metrics
    # compute overall PBO across folds
    pbo = probability_of_backtest_overfitting(train_sharpes, test_sharpes)
    # update best metrics with global PBO
    if best_metrics is not None:
        best_metrics.pbo = pbo
    return best_metrics if best_metrics else StrategyMetrics(0, 0, 0, 0, 1.0, 0, 0)