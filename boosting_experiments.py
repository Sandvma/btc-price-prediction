"""Run gradient boosting experiments on BTC-USD returns.

This script explores more powerful ensemble models—XGBoost, LightGBM and
CatBoost—to forecast logarithmic returns of Bitcoin at multiple prediction
horizons. These models are tree-based and can capture non‑linear patterns
without requiring heavy deep‑learning frameworks, making them a practical
choice in lightweight environments. The script computes a set of technical
indicators (RSI, MACD, Bollinger band distances and ATR) along with lagged
returns to form the feature matrix. For each prediction horizon (1, 3, 5
and 10 days ahead), the data are split chronologically into training and
testing sets (80/20 split). Each regressor is trained on the training
portion and evaluated on the held‑out set using mean squared error and
directional accuracy (percentage of times the sign of the prediction
matches the actual return). Results are written to ``boosting_results.txt``.

Usage:
    python boosting_experiments.py

The output file ``boosting_results.txt`` will contain a tabular summary of
metrics for each model and horizon.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and lagged returns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'Open', 'High', 'Low', 'Close', 'Adj Close'.

    Returns
    -------
    features : pandas.DataFrame
        DataFrame of computed features aligned with returns.
    returns : pandas.Series
        Logarithmic returns of the adjusted close price.
    """
    # Choose adjusted close if available else close
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    returns = np.log(close).diff().fillna(0)

    # RSI (14 period) using simple moving averages
    window_rsi = 14
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window_rsi).mean()
    roll_down = down.rolling(window_rsi).mean()
    # Avoid division by zero
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))

    # MACD and signal line (EMAs)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger bands (20 period)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_high = sma20 + 2 * std20
    bb_low = sma20 - 2 * std20
    bb_dist_high = (close - bb_high) / close
    bb_dist_low = (close - bb_low) / close

    # Average True Range (ATR, 14 period)
    high = df['High']
    low = df['Low']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # Lagged returns (1–5 days)
    max_lag = 5
    lags = {f'lag_{i}': returns.shift(i) for i in range(1, max_lag + 1)}

    # Assemble features
    features = pd.DataFrame({
        'rsi14': rsi,
        'macd': macd_line,
        'macd_signal': macd_signal,
        'bb_dist_high': bb_dist_high,
        'bb_dist_low': bb_dist_low,
        'atr': atr
    }).assign(**lags)

    # Drop rows with NaNs caused by indicator calculations
    features = features.dropna()
    returns = returns.loc[features.index]
    return features, returns


def run_experiments(path: str, horizons: list[int]) -> list[tuple]:
    """Train boosting models and evaluate them on multiple horizons.

    Parameters
    ----------
    path : str
        Path to the CSV file containing BTC-USD data with OHLCV columns.
    horizons : list of int
        List of forecast horizons (in days) to evaluate.

    Returns
    -------
    results : list of tuples
        Each tuple contains horizon and metrics for XGBoost, LightGBM and CatBoost.
    """
    df = pd.read_csv(path)
    features, returns = compute_indicators(df)

    results = []
    for h in horizons:
        # Create horizon target by shifting returns negatively
        target = returns.shift(-h).loc[features.index]
        # Align data by dropping NaNs at end
        valid_idx = target.dropna().index
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        # Chronological train/test split (80/20)
        n = len(X)
        split_idx = int(n * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Standardise numerical features for XGBoost only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='rmse',
            verbosity=0,
        )
        xgb_model.fit(X_train_scaled, y_train)
        pred_xgb = xgb_model.predict(X_test_scaled)
        mse_xgb = mean_squared_error(y_test, pred_xgb)
        dir_acc_xgb = (np.sign(pred_xgb) == np.sign(y_test)).mean()

        # Train LightGBM regressor (works directly on dataframe)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            objective='regression'
        )
        lgb_model.fit(X_train, y_train)
        pred_lgb = lgb_model.predict(X_test)
        mse_lgb = mean_squared_error(y_test, pred_lgb)
        dir_acc_lgb = (np.sign(pred_lgb) == np.sign(y_test)).mean()

        # Train CatBoost regressor (catboost handles categorical features natively but we have none)
        cat_model = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        pred_cat = cat_model.predict(X_test)
        mse_cat = mean_squared_error(y_test, pred_cat)
        dir_acc_cat = (np.sign(pred_cat) == np.sign(y_test)).mean()

        results.append((h, mse_xgb, dir_acc_xgb, mse_lgb, dir_acc_lgb, mse_cat, dir_acc_cat))
    return results


def save_results(results: list[tuple], output_file: str) -> None:
    """Write experiment results to a text file.

    Parameters
    ----------
    results : list of tuples
        Results produced by ``run_experiments``.
    output_file : str
        Path to the output text file.
    """
    header = (
        "Horizon\tXGB_MSE\tXGB_DirAcc\tLGB_MSE\tLGB_DirAcc"
        "\tCAT_MSE\tCAT_DirAcc"
    )
    lines = [header]
    for r in results:
        line = (
            f"{r[0]}\t{r[1]:.6f}\t{r[2]:.4f}\t"
            f"{r[3]:.6f}\t{r[4]:.4f}\t{r[5]:.6f}\t{r[6]:.4f}"
        )
        lines.append(line)
    Path(output_file).write_text("\n".join(lines))


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'BTC-USD.csv')
    horizons = [1, 3, 5, 10]
    results = run_experiments(data_path, horizons)
    output_path = os.path.join(os.path.dirname(__file__), 'boosting_results.txt')
    save_results(results, output_path)
    print(f"Results saved to {output_path}")
    # Also print to console
    print("Horizon\tXGB_MSE\tXGB_DirAcc\tLGB_MSE\tLGB_DirAcc\tCAT_MSE\tCAT_DirAcc")
    for r in results:
        print(
            f"{r[0]}\t{r[1]:.6f}\t{r[2]:.4f}\t"
            f"{r[3]:.6f}\t{r[4]:.4f}\t{r[5]:.6f}\t{r[6]:.4f}"
        )


if __name__ == '__main__':
    main()