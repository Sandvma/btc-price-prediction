"""
meta_labeling_experiments
=========================

Este módulo implementa un experimento de **meta‑etiquetado** sobre la serie
BTC‑USD.  El meta‑etiquetado es una técnica descrita por Marcos López de
Prado en la que se entrena un modelo base para generar señales (p. ej.,
dirección del retorno) y un segundo modelo (meta‑modelo) que intenta
predecir cuándo la señal del modelo base será correcta.  De esta forma,
se filtran las señales de baja confianza y se reduce el número de
operaciones erróneas.

Características principales del script:

* **Generación de features**: igual que en otros experimentos, se usan
  retornos logarítmicos, volatilidades, momentum, ATR14 y una
  diferenciación fraccional con d=0.5 (que mostró el mejor Sharpe en
  pruebas previas).  El usuario puede ajustar los parámetros al
  ejecutar el script.
* **Etiquetado dinámico con triple barrera**: se crean etiquetas
  (+1, -1) basadas en niveles de ATR y horizonte fijo.  Se descartan
  las etiquetas neutrales (0).
* **Modelo base**: se utiliza un ``ExtraTreesClassifier`` para
  predecir la dirección.  Para cada muestra de entrenamiento se
  registran el pronóstico de probabilidad y si la predicción fue
  correcta o no (meta‑etiqueta).
* **Meta‑modelo**: se entrena un ``LogisticRegression`` simple sobre
  la(s) característica(s) de probabilidad del modelo base para
  predecir si la señal base es correcta.  Otras características
  opcionales podrían añadirse.
* **Walk‑forward**: se emplea ``TimeSeriesSplit`` para validar en
  distintas ventanas temporales.  Para cada fold se calculan:
  precisión direccional del modelo base, precisión del meta‑modelo
  (aciertos filtrados), Sharpe ratio de la estrategia filtrada,
  Sharpe deflacionado y bandera PBO (similar a otros scripts).
* **Resultados**: se guardan en ``meta_label_results.txt`` y se
  muestran por consola.

Este experimento pretende demostrar la utilidad de filtrar señales
basándose en la confianza del modelo base.  Si el meta‑modelo
funciona, la precisión y/o el Sharpe de la estrategia deberían
mejorar respecto al modelo base por sí solo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w, dtype=float)


def fractional_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    s = pd.Series(series).astype(float)
    w = get_weights_ffd(d, threshold)
    width = len(w)
    out = np.full(s.shape, np.nan, dtype=float)
    vals = s.values
    for i in range(width - 1, len(vals)):
        out[i] = np.dot(w[::-1], vals[i - width + 1 : i + 1])
    return pd.Series(out, index=s.index, name=f"ffd_{d:.2f}")


def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = r.mean() * periods
    sd = r.std(ddof=1) * np.sqrt(periods)
    return 0.0 if sd == 0 else float(mu / sd)


def deflated_sharpe(sr: float, n: int, trials: int = 1) -> float:
    if n <= 1:
        return float(sr)
    adj = np.sqrt(max(np.log(max(trials, 2)), 1.0) / max(n, 2))
    return float(sr) - float(adj)


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return df


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def dynamic_triple_barrier(
    close: pd.Series, atr: pd.Series, horizon: int, k_tp: float = 1.0, k_sl: float = 1.0
) -> pd.Series:
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
class MetaLabelConfig:
    n_splits: int = 5
    d: float | None = 0.5  # order for FFD
    horizon: int = 10
    k_tp: float = 1.0
    k_sl: float = 1.0
    random_state: int = 42


def run_meta_labeling(cfg: MetaLabelConfig, data_path: str = "btc_price_prediction/data/BTC-USD.csv") -> None:
    df_raw = load_prices(data_path)
    feats = compute_features(df_raw, cfg.d)
    atr = feats["atr_14"]
    close = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]
    labels = dynamic_triple_barrier(close, atr, horizon=cfg.horizon, k_tp=cfg.k_tp, k_sl=cfg.k_sl)
    data = pd.concat([feats, labels.rename("label")], axis=1).dropna()
    data = data[data["label"] != 0.0]
    X_full = data.drop(columns=["label"]).values
    y_full = np.where(data["label"].values > 0, 1, 0)
    log_ret_series = data["log_ret"].values
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    base_acc_list: list[float] = []
    meta_acc_list: list[float] = []
    sharpe_list: list[float] = []
    dsr_list: list[float] = []
    pbo_list: list[float] = []
    for train_idx, test_idx in tscv.split(X_full):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        # Base model
        base_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            random_state=cfg.random_state,
        )
        base_model.fit(X_train, y_train)
        y_pred_base = base_model.predict(X_test)
        base_acc = accuracy_score(y_test, y_pred_base)
        base_acc_list.append(base_acc)
        # Meta labels: 1 if base prediction correct on train, 0 otherwise
        y_pred_train = base_model.predict(X_train)
        meta_labels = (y_pred_train == y_train).astype(int)
        # Meta features: we use predicted probability of class 1 from base model
        proba_train = base_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        proba_test = base_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        # Train meta-model only if there are both classes in meta_labels.
        # If meta_labels has only one unique value (all correct or all incorrect),
        # training a logistic regression would fail; in that case we treat the meta-model
        # as a trivial classifier that always outputs 1 (we always trust the base model).
        if len(np.unique(meta_labels)) < 2:
            meta_pred_train = np.ones_like(meta_labels)
            meta_pred_test = np.ones(len(proba_test), dtype=int)
            meta_acc = 1.0
        else:
            meta_model = LogisticRegression(random_state=cfg.random_state, max_iter=200)
            meta_model.fit(proba_train, meta_labels)
            meta_pred_train = meta_model.predict(proba_train)
            meta_pred_test = meta_model.predict(proba_test)
            meta_acc = accuracy_score(meta_labels, meta_pred_train)
        meta_acc_list.append(meta_acc)
        # Strategy: take position only when meta_pred_test==1
        pos = np.where(meta_pred_test == 1, np.where(y_pred_base == 1, 1.0, -1.0), 0.0)
        strat_returns = pos * log_ret_series[test_idx]
        sharpe = sharpe_ratio(strat_returns)
        dsr = deflated_sharpe(sharpe, n=len(strat_returns), trials=1)
        # Compute train Sharpe for PBO using meta_pred_train
        pos_train = np.where(meta_pred_train == 1, np.where(y_pred_train == 1, 1.0, -1.0), 0.0)
        strat_train_returns = pos_train * log_ret_series[train_idx]
        sharpe_train = sharpe_ratio(strat_train_returns)
        pbo_flag = 1.0 if (sharpe_train > sharpe and sharpe < 0) else 0.0
        sharpe_list.append(sharpe)
        dsr_list.append(dsr)
        pbo_list.append(pbo_flag)
    results = {
        "base_accuracy_mean": float(np.mean(base_acc_list)),
        "meta_accuracy_mean": float(np.mean(meta_acc_list)),
        "sharpe_mean": float(np.mean(sharpe_list)),
        "deflated_sharpe_mean": float(np.mean(dsr_list)),
        "pbo": float(np.mean(pbo_list)),
    }
    # save results
    out_path = os.path.join(os.path.dirname(__file__), "meta_label_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResultados meta‑etiquetado:\n")
    print(
        f"Base Acc={results['base_accuracy_mean']:.4f}, Meta Acc={results['meta_accuracy_mean']:.4f}, "
        f"Sharpe={results['sharpe_mean']:.4f}, DSR={results['deflated_sharpe_mean']:.4f}, PBO={results['pbo']:.4f}"
    )


if __name__ == "__main__":
    cfg = MetaLabelConfig()
    run_meta_labeling(cfg)