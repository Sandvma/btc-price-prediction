"""
walk_forward_robust
====================

Este módulo implementa una validación **walk‑forward** más rigurosa
para la serie BTC‑USD.  Consta de los siguientes componentes:

* **Generación de características**: se incluyen rendimientos logarítmicos,
  medidas de volatilidad (5 y 10 días), momentos (5 y 10 días), el ATR14
  y una lista de diferenciaciones fraccionales a varios órdenes (d ∈ {None,
  0.3, 0.4, 0.5, 0.6}).  Esto permite evaluar si la estacionariedad inducida
  por la FFD mejora las métricas de trading.
* **Etiquetado con triple barrera dinámica**: los niveles de take‑profit y
  stop‑loss se definen como múltiplos del ATR; se emplea un horizonte
  configurable para buscar el evento que se active primero y se descartan
  etiquetas neutras (0) para centrarse en señales direccionales.
* **Esquema walk‑forward**: utiliza ``TimeSeriesSplit`` de ``scikit‑learn``
  con 5 particiones, cada una funcionando como una ventana deslizante.  Tras
  cada entrenamiento se calcula la precisión direccional, el ratio de Sharpe
  anualizado, el Sharpe deflacionado (DSR) y una aproximación sencilla de
  PBO (probabilidad de sobreajuste) comparando los Sharpes de entrenamiento
  y de validación.
* **Resultados**: se escriben en ``walk_forward_results.txt`` y se
  imprimen en consola para facilitar su análisis.  El archivo de resultados
  contiene un diccionario en formato JSON con las métricas promedio por
  cada orden de fraccionalización.

Para ejecutar el script desde la raíz del repositorio:

```
python -m btc_price_prediction.walk_forward_robust
```

El objetivo de este experimento es servir como base para análisis más
avanzados (optimización de hiperparámetros, meta‑etiquetado, modelos
condicionados por regímenes, etc.), ofreciendo una evaluación
robusta y reproducible de la calidad de las señales generadas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    """Compute fractional differentiation weights up to a threshold.

    Parameters
    ----------
    d: float
        Fractional differencing order.
    threshold: float
        Cut‑off threshold for the smallest weight magnitude.
    max_size: int
        Maximum number of coefficients to compute.

    Returns
    -------
    np.ndarray
        Array of weights (not reversed).
    """
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w, dtype=float)


def fractional_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Apply fixed‑width fractional differentiation (FFD) to a price series.

    Uses the methodology from López de Prado's AFML to produce a series
    that retains memory at all horizons while achieving stationarity.
    """
    s = pd.Series(series).astype(float)
    w = get_weights_ffd(d, threshold)
    width = len(w)
    out = np.full(s.shape, np.nan, dtype=float)
    vals = s.values
    for i in range(width - 1, len(vals)):
        out[i] = np.dot(w[::-1], vals[i - width + 1:i + 1])
    return pd.Series(out, index=s.index, name=f"ffd_{d:.2f}")


def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """Compute annualised Sharpe ratio of returns array."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = r.mean() * periods
    sd = r.std(ddof=1) * np.sqrt(periods)
    return 0.0 if sd == 0 else float(mu / sd)


def deflated_sharpe(sr: float, n: int, trials: int = 1) -> float:
    """Approximate deflated Sharpe ratio.

    Adjusts the Sharpe ratio downward based on the number of observations and
    the number of trials (models) considered to mitigate selection bias.
    """
    if n <= 1:
        return float(sr)
    adj = np.sqrt(max(np.log(max(trials, 2)), 1.0) / max(n, 2))
    return float(sr) - float(adj)


def load_prices(path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV and sort by date."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return df


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) over n periods."""
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def dynamic_triple_barrier(
    close: pd.Series, atr: pd.Series, horizon: int, k_tp: float = 1.0, k_sl: float = 1.0
) -> pd.Series:
    """Compute dynamic triple barrier labels based on ATR thresholds.

    For each timestamp i, uses horizon h bars ahead.  A take‑profit event
    occurs if the future price crosses above price[i] + k_tp * atr[i]; a
    stop‑loss event occurs if it crosses below price[i] - k_sl * atr[i].
    The label is +1 if TP is hit first, -1 if SL is hit first, and 0 if
    neither threshold is hit within the horizon.
    """
    n = len(close)
    labels = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(n - horizon):
        price = close.iloc[i]
        atr_i = atr.iloc[i]
        if pd.isna(atr_i):
            labels.iloc[i] = np.nan
            continue
        tp_price = price + k_tp * atr_i
        sl_price = price - k_sl * atr_i
        window = close.iloc[i + 1 : i + horizon + 1]
        tp_hit_idx = window[window >= tp_price].index.min() if any(window >= tp_price) else None
        sl_hit_idx = window[window <= sl_price].index.min() if any(window <= sl_price) else None
        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx <= sl_hit_idx):
            labels.iloc[i] = 1.0
        elif sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx):
            labels.iloc[i] = -1.0
        else:
            labels.iloc[i] = 0.0
    return labels


def compute_features(df: pd.DataFrame, d: float | None) -> pd.DataFrame:
    """Compute feature set for the model.

    Features: log return, volatility (5,10), momentum (5,10), ATR14 and
    fractional differentiation of order ``d`` (if not None).
    """
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    high = df["High"]
    low = df["Low"]
    df_feat = pd.DataFrame(index=df.index)
    df_feat["log_ret"] = np.log(close / close.shift(1))
    df_feat["vol_5"] = df_feat["log_ret"].rolling(window=5).std()
    df_feat["vol_10"] = df_feat["log_ret"].rolling(window=10).std()
    df_feat["mom_5"] = df_feat["log_ret"].rolling(window=5).mean()
    df_feat["mom_10"] = df_feat["log_ret"].rolling(window=10).mean()
    df_feat["atr_14"] = compute_atr(high, low, close, n=14)
    if d is not None:
        ffd_series = fractional_diff_ffd(close, d=d, threshold=1e-5)
        df_feat[f"ffd_{d:.2f}"] = ffd_series
    return df_feat


@dataclass
class WalkForwardConfig:
    """Configuration for walk‑forward experiment."""
    # Use 3 splits to accommodate cases with few samples (e.g., higher d values reduce data).
    n_splits: int = 3
    # Try a grid of differentiation orders.  None corresponds to no FFD.
    #
    # Note: fractional differencing with very small d (e.g., 0.1 or 0.2) can
    # produce so many NaNs that no labelled samples remain after the triple
    # barrier filter, causing cross‑validation to fail.  We therefore omit
    # values that typically yield an empty dataset (e.g., 0.2).
    d_values: tuple[float | None, ...] = (
        None,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    )
    horizon: int = 10
    k_tp: float = 1.0
    k_sl: float = 1.0
    random_state: int = 42


def run_experiment(cfg: WalkForwardConfig, data_path: str = "btc_price_prediction/data/BTC-USD.csv") -> None:
    """Run a walk‑forward experiment across different fractional differentiation orders.

    Parameters
    ----------
    cfg: WalkForwardConfig
        Configuration with hyperparameters.
    data_path: str
        Path to CSV file containing OHLCV data.
    """
    df_raw = load_prices(data_path)
    results: dict[str, dict[str, float]] = {}
    for d in cfg.d_values:
        feats = compute_features(df_raw, d)
        atr = feats["atr_14"]
        close = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]
        labels = dynamic_triple_barrier(close, atr, horizon=cfg.horizon, k_tp=cfg.k_tp, k_sl=cfg.k_sl)
        data = pd.concat([feats, labels.rename("label")], axis=1).dropna()
        data = data[data["label"] != 0.0]  # remove neutral events
        # Skip this differentiation order if not enough samples remain for CV
        if len(data) < (cfg.n_splits + 1):
            # Record NaNs to indicate insufficient data
            results[str(d)] = {
                "accuracy_mean": float("nan"),
                "sharpe_mean": float("nan"),
                "deflated_sharpe_mean": float("nan"),
                "pbo": float("nan"),
            }
            continue
        X = data.drop(columns=["label"]).values
        y = np.where(data["label"].values > 0, 1, 0)  # binarise labels
        tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
        acc_scores: list[float] = []
        sharpe_scores: list[float] = []
        dsr_scores: list[float] = []
        pbo_flags: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=None,
                max_features="sqrt",
                random_state=cfg.random_state,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            pos = np.where(y_pred == 1, 1.0, -1.0)
            log_ret_series = data["log_ret"].values
            strat_returns = pos * log_ret_series[test_idx]
            sharpe = sharpe_ratio(strat_returns)
            dsr = deflated_sharpe(sharpe, n=len(strat_returns), trials=len(cfg.d_values))
            pos_train = np.where(model.predict(X_train) == 1, 1.0, -1.0)
            strat_train_returns = pos_train * log_ret_series[train_idx]
            sharpe_train = sharpe_ratio(strat_train_returns)
            # PBO flag: 1 if train Sharpe > test Sharpe and test Sharpe < 0 (sign of overfitting)
            pbo_flag = 1.0 if (sharpe_train > sharpe and sharpe < 0) else 0.0
            acc_scores.append(acc)
            sharpe_scores.append(sharpe)
            dsr_scores.append(dsr)
            pbo_flags.append(pbo_flag)
        results[str(d)] = {
            "accuracy_mean": float(np.mean(acc_scores)),
            "sharpe_mean": float(np.mean(sharpe_scores)),
            "deflated_sharpe_mean": float(np.mean(dsr_scores)),
            "pbo": float(np.mean(pbo_flags)),
        }
    # save results
    out_path = os.path.join(os.path.dirname(__file__), "walk_forward_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # print summary
    print("\nResultados walk‑forward robusto:\n")
    for d, metrics in results.items():
        print(
            f"d={d}: Accuracy={metrics['accuracy_mean']:.4f}, "
            f"Sharpe={metrics['sharpe_mean']:.4f}, "
            f"DSR={metrics['deflated_sharpe_mean']:.4f}, PBO={metrics['pbo']:.4f}"
        )


if __name__ == "__main__":
    cfg = WalkForwardConfig()
    run_experiment(cfg)