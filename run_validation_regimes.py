"""
Script de validación robusta con CPCV y modelos condicionados por regímenes.

Este script carga el conjunto de datos BTC‑USD, genera un conjunto de
características de retorno y volatilidad, etiqueta los retornos futuros
como clase binaria, y a continuación evalúa un modelo global y un
conjunto de modelos condicionados por regímenes (detectados mediante
K‑Means) usando Combinatorial Purged Cross‑Validation (CPCV) con
embargo.  Las métricas calculadas incluyen precisión direccional,
retorno acumulado, ratio de Sharpe, Sharpe deflacionado y PBO para
comparar el riesgo de sobreoptimización entre el modelo global y el
modelo condicionado por regímenes.  Los resultados se guardan en
``config/validation_regimes.results.json``.

Uso:

    python run_validation_regimes.py

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from robust.validation import iter_cpcv
from robust.metrics import sharpe_ratio, deflated_sharpe, pbo_two_candidates
from regimes.model import fit_kmeans


def load_dataset(path: str) -> pd.DataFrame:
    """Carga datos OHLC y calcula columnas de ATR.

    Args:
        path: Ruta al CSV con columnas Date, Open, High, Low, Close, Adj Close.

    Returns:
        DataFrame con columnas originales y ATR14 (True Range media sobre 14 días).
    """
    df = pd.read_csv(path, parse_dates=[0])
    df = df.sort_values('Date').reset_index(drop=True)
    # Calcular true range
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR 14
    atr14 = true_range.rolling(window=14).mean().fillna(method='bfill')
    df['ATR14'] = atr14
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construye características y etiqueta binaria a partir del DataFrame.

    Características:
      - ret_1: retorno diario (pct_change) del cierre ajustado
      - ret_20: retorno a 20 días
      - vol_20: desviación estándar de retornos a 20 días
      - vol_60: desviación estándar de retornos a 60 días
      - atr_ratio: ATR14 dividido por Close

    Etiqueta: y[t] = 1 si ret_1[t+1] > 0, 0 en caso contrario.  El último
    valor se descarta ya que no tiene retorno futuro.

    Args:
        df: DataFrame con columnas Date, Close, ATR14, etc.

    Returns:
        (X, y) donde X es un DataFrame de características y y un Serie de etiquetas.
    """
    adj_close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    ret_1 = adj_close.pct_change().fillna(0.0)
    ret_20 = adj_close.pct_change(periods=20).fillna(0.0)
    # Volatilidad de retornos
    vol_20 = ret_1.rolling(window=20).std().fillna(method='bfill')
    vol_60 = ret_1.rolling(window=60).std().fillna(method='bfill')
    atr_ratio = df['ATR14'] / df['Close']
    # Construir X sin la última fila porque y será ret_1.shift(-1)
    X = pd.DataFrame({
        'ret_1': ret_1,
        'ret_20': ret_20,
        'vol_20': vol_20,
        'vol_60': vol_60,
        'atr_ratio': atr_ratio,
    })
    # Etiqueta: retorno de mañana
    y = (ret_1.shift(-1) > 0).astype(int)
    # Alinear
    X = X.iloc[:-1, :].reset_index(drop=True)
    y = y.iloc[:-1].reset_index(drop=True)
    return X, y


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    K: int = 6,
    m: int = 2,
    embargo: int = 5,
    n_clusters: int = 3,
    min_cluster_samples: int = 20,
    random_state: int = 0,
) -> Dict[str, float]:
    """Ejecuta CPCV con modelos global y por régimen y calcula métricas.

    Args:
        X: Matriz de características.
        y: Vector binario de etiquetas.
        K: Número de bloques para CPCV.
        m: Número de bloques usados como validación en cada fold.
        embargo: Tamaño del embargo temporal.
        n_clusters: Número de regímenes (clusters) para K‑Means.
        min_cluster_samples: Mínimo de muestras para entrenar un modelo de
            régimen; si no se alcanza, se usa el modelo global.
        random_state: Semilla para reproducibilidad.

    Returns:
        Diccionario con métricas agregadas y PBO.
    """
    n = len(X)
    global_accs = []
    regime_accs = []
    global_sharpes = []
    regime_sharpes = []
    in_sample_metrics = []
    out_sample_metrics = []
    # Convert to numpy for speed
    X_np = X.values
    y_np = y.values
    fold = 0
    for train_idx, val_idx in iter_cpcv(n, K, m, embargo):
        fold += 1
        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]
        # Entrenar modelo global.  Usar menos árboles para acelerar la evaluación
        global_model = ExtraTreesClassifier(n_estimators=50, random_state=random_state)
        global_model.fit(X_train, y_train)
        # Predicción in‑sample global
        pred_train_global = global_model.predict(X_train)
        acc_train_global = accuracy_score(y_train, pred_train_global)
        # Predicción out‑sample global
        pred_val_global = global_model.predict(X_val)
        acc_val_global = accuracy_score(y_val, pred_val_global)
        # Estrategia global: long cuando pred=1, short cuando pred=0
        ret_val = y.shift(-1).fillna(0.0).values  # next-day return for full series
        # Map predictions to positions: 1 -> long, 0 -> short
        pos_global = np.where(pred_val_global == 1, 1, -1)
        # returns for val_idx+1 shift; we must align indices: ret_val[val_idx] is current day's return; we need ret_val[val_idx]
        ret_use = ret_val[val_idx]
        strat_ret_global = pos_global * ret_use
        sharpe_global = sharpe_ratio(strat_ret_global)
        # entrenar KMeans en X_train para regímenes
        regime_model = fit_kmeans(X_train, n_clusters=n_clusters, random_state=random_state)
        train_labels = regime_model.predict(X_train)
        # Entrenar un modelo por régimen
        cluster_models = {}
        for cluster_id in np.unique(train_labels):
            cluster_idx = np.where(train_labels == cluster_id)[0]
            if len(cluster_idx) >= min_cluster_samples:
                # Entrenar con menos árboles para acelerar
                clf = ExtraTreesClassifier(n_estimators=50, random_state=random_state)
                clf.fit(X_train[cluster_idx], y_train[cluster_idx])
                cluster_models[cluster_id] = clf
        # Predicción in‑sample para régimen
        pred_train_regime = []
        for xi in range(len(X_train)):
            cl = train_labels[xi]
            if cl in cluster_models:
                pred_train_regime.append(cluster_models[cl].predict(X_train[xi:xi+1])[0])
            else:
                pred_train_regime.append(global_model.predict(X_train[xi:xi+1])[0])
        acc_train_regime = accuracy_score(y_train, pred_train_regime)
        # Predicción out‑sample para régimen
        val_labels = regime_model.predict(X_val)
        pred_val_regime = []
        for xi, cl in enumerate(val_labels):
            if cl in cluster_models:
                pred_val_regime.append(cluster_models[cl].predict(X_val[xi:xi+1])[0])
            else:
                pred_val_regime.append(global_model.predict(X_val[xi:xi+1])[0])
        acc_val_regime = accuracy_score(y_val, pred_val_regime)
        # estrategia por régimen
        pos_regime = np.where(np.array(pred_val_regime) == 1, 1, -1)
        strat_ret_regime = pos_regime * ret_use
        sharpe_regime = sharpe_ratio(strat_ret_regime)
        # registrar métricas
        global_accs.append(acc_val_global)
        regime_accs.append(acc_val_regime)
        global_sharpes.append(sharpe_global)
        regime_sharpes.append(sharpe_regime)
        # almacenar métricas para PBO: usar accuracy in‑sample y out‑sample
        in_sample_metrics.append((acc_train_global, acc_train_regime))
        out_sample_metrics.append((acc_val_global, acc_val_regime))
    # Agregados
    results: Dict[str, float] = {}
    results['global_accuracy'] = float(np.mean(global_accs))
    results['regime_accuracy'] = float(np.mean(regime_accs))
    # calcular retornos agregados concatenando los strat_ret? mejor sumarlos
    results['global_sharpe'] = float(np.mean(global_sharpes))
    results['regime_sharpe'] = float(np.mean(regime_sharpes))
    # Sharpe deflacionado: usar número de folds y 2 candidatos
    folds = len(global_accs)
    results['global_deflated_sharpe'] = float(deflated_sharpe(np.mean(global_sharpes), folds, trials=2))
    results['regime_deflated_sharpe'] = float(deflated_sharpe(np.mean(regime_sharpes), folds, trials=2))
    # PBO
    results['pbo'] = float(pbo_two_candidates(in_sample_metrics, out_sample_metrics))
    return results


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data' / 'BTC-USD.csv'
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    df = load_dataset(str(data_path))
    X, y = build_features(df)
    results = evaluate_models(X, y, K=6, m=2, embargo=5, n_clusters=3, min_cluster_samples=20, random_state=0)
    # Escribir resultados
    out_path = base_dir / 'config' / 'validation_regimes.results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Resultados CPCV + regímenes:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()