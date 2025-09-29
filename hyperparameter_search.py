"""
hyperparameter_search
======================

Este módulo ejecuta una búsqueda en rejilla (grid search) simplificada
sobre distintos parámetros de la validación walk‑forward para la serie
BTC‑USD.  El objetivo es identificar combinaciones de horizonte
(``horizon``), multiplicadores de ATR para take‑profit/stop‑loss
(``k_tp``/``k_sl``) y orden de diferenciación fraccional (``d``)
que maximicen métricas de rendimiento fuera de muestra como el
Sharpe y el Sharpe deflacionado.

Resumen de funcionalidades:

* **Features:** se calculan retornos logarítmicos, volatilidades
  (5 y 10 días), momentum (5 y 10 días), ATR14 y, si corresponde,
  la serie fraccionada por diferenciación fraccional (FFD) de orden
  ``d``.  Las funciones ``compute_features`` y ``dynamic_triple_barrier``
  se reutilizan del módulo ``walk_forward_robust``.
* **Etiquetado:** utiliza la triple barrera dinámica donde los niveles
  de TP y SL se fijan como ``k_tp * ATR`` y ``k_sl * ATR``.  Solo se
  consideran las etiquetas +1 (TP) y ‑1 (SL); las neutras se descartan.
* **Validación walk‑forward:** implementa ``TimeSeriesSplit`` con
  tres particiones (n_splits=3) para respetar la estructura temporal.
  Para cada fold se entrena un ``ExtraTreesClassifier`` y se calcula
  precisión, Sharpe anualizado, Sharpe deflacionado y PBO
  (probabilidad de sobreajuste simple).
* **Búsqueda de hiperparámetros:** recorre las listas definidas en
  ``HORIZON_GRID``, ``K_GRID`` y ``D_GRID``.  Los resultados se
  almacenan en una lista y se escriben en ``hyperparameter_results.txt``.

Para ejecutar la búsqueda desde la raíz del repositorio:

```
python -m btc_price_prediction.hyperparameter_search
```

Los resultados se imprimen por pantalla ordenados por Sharpe
deflacionado (de mayor a menor) y se guardan en un archivo JSON
para posterior análisis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Reutilizamos funciones del módulo walk_forward_robust para evitar duplicar lógica
from .walk_forward_robust import (
    compute_features,
    dynamic_triple_barrier,
    sharpe_ratio,
    deflated_sharpe,
    load_prices,
)


# Grids de búsqueda.  Se pueden ampliar si se desea explorar más valores
HORIZON_GRID: List[int] = [3, 5, 10]
K_GRID: List[float] = [0.5, 0.7, 1.0, 1.5, 2.0]
D_GRID: List[float | None] = [0.5, 0.6]  # None significa sin FFD


@dataclass
class SearchResult:
    horizon: int
    k: float
    d: float | None
    accuracy: float
    sharpe: float
    dsr: float
    pbo: float
    samples: int


def evaluate_combination(
    df_raw: pd.DataFrame,
    horizon: int,
    k: float,
    d: float | None,
    n_splits: int = 3,
    n_estimators: int = 100,
    random_state: int = 42,
) -> SearchResult | None:
    """Evaluar una combinación de hiperparámetros.

    Parameters
    ----------
    df_raw: pd.DataFrame
        Serie OHLCV ya cargada.
    horizon: int
        Número de barras hacia adelante para el horizonte de predicción.
    k: float
        Multiplicador de ATR para TP y SL.
    d: float | None
        Orden de diferenciación fraccional (None = sin FFD).
    n_splits: int
        Número de splits en TimeSeriesSplit.
    n_estimators: int
        Número de árboles en ExtraTrees.
    random_state: int
        Semilla aleatoria para reproducibilidad.

    Returns
    -------
    SearchResult | None
        Objeto con métricas.  Devuelve None si no hay suficientes muestras.
    """
    close = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]
    feats = compute_features(df_raw, d)
    atr = feats["atr_14"]
    labels = dynamic_triple_barrier(close, atr, horizon=horizon, k_tp=k, k_sl=k)
    data = pd.concat([feats, labels.rename("label")], axis=1).dropna()
    data = data[data["label"] != 0.0]
    # Se necesitan al menos (n_splits + 1) muestras para ejecutar CV
    if len(data) < (n_splits + 1):
        return None
    X = data.drop(columns=["label"]).values
    y = np.where(data["label"].values > 0, 1, 0)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_scores: List[float] = []
    sharpe_scores: List[float] = []
    dsr_scores: List[float] = []
    pbo_flags: List[float] = []
    log_ret_series = data["log_ret"].values
    for train_idx, test_idx in tscv.split(X):
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            random_state=random_state,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], y_pred))
        # calcular retornos de la estrategia (posición = +1 si predicción=1, -1 si predicción=0)
        pos_test = np.where(y_pred == 1, 1.0, -1.0)
        returns_test = pos_test * log_ret_series[test_idx]
        sharpe = sharpe_ratio(returns_test)
        dsr = deflated_sharpe(sharpe, n=len(returns_test), trials=len(D_GRID) * len(K_GRID) * len(HORIZON_GRID))
        sharpe_scores.append(sharpe)
        dsr_scores.append(dsr)
        # PBO: si el Sharpe en entrenamiento es mayor que en test y el test es negativo, indica sobreajuste
        pos_train = np.where(model.predict(X[train_idx]) == 1, 1.0, -1.0)
        returns_train = pos_train * log_ret_series[train_idx]
        sharpe_train = sharpe_ratio(returns_train)
        pbo_flags.append(1.0 if (sharpe_train > sharpe and sharpe < 0) else 0.0)
    return SearchResult(
        horizon=horizon,
        k=k,
        d=d,
        accuracy=float(np.mean(acc_scores)),
        sharpe=float(np.mean(sharpe_scores)),
        dsr=float(np.mean(dsr_scores)),
        pbo=float(np.mean(pbo_flags)),
        samples=len(data),
    )


def run_search(data_path: str = "btc_price_prediction/data/BTC-USD.csv") -> List[SearchResult]:
    """Ejecutar búsqueda en rejilla sobre las combinaciones definidas.

    Parameters
    ----------
    data_path: str
        Ruta al CSV de datos OHLCV.

    Returns
    -------
    List[SearchResult]
        Lista de resultados ordenada por DSr descendente.
    """
    df_raw = load_prices(data_path)
    results: List[SearchResult] = []
    for horizon in HORIZON_GRID:
        for k in K_GRID:
            for d in D_GRID:
                res = evaluate_combination(df_raw, horizon=horizon, k=k, d=d)
                if res is not None:
                    results.append(res)
    # ordenar por DSr descendente, luego Sharpe y luego precisión
    results.sort(key=lambda x: (-x.dsr, -x.sharpe, -x.accuracy))
    return results


def save_results(results: List[SearchResult], out_path: str) -> None:
    """Guardar resultados en archivo JSON.

    Parameters
    ----------
    results: List[SearchResult]
        Lista de resultados a guardar.
    out_path: str
        Ruta del archivo donde se guardará el JSON.
    """
    serializable: List[Dict[str, Any]] = [
        {
            "horizon": r.horizon,
            "k": r.k,
            "d": r.d,
            "accuracy": r.accuracy,
            "sharpe": r.sharpe,
            "deflated_sharpe": r.dsr,
            "pbo": r.pbo,
            "samples": r.samples,
        }
        for r in results
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def main() -> None:
    results = run_search()
    # Ruta para guardar los resultados junto a este script
    out_path = os.path.join(os.path.dirname(__file__), "hyperparameter_results.txt")
    save_results(results, out_path)
    print("\nResultados de la búsqueda de hiperparámetros (ordenados por DSr):\n")
    for r in results[:10]:  # muestra los 10 mejores
        print(
            f"h={r.horizon}, k={r.k}, d={r.d}: Accuracy={r.accuracy:.4f}, Sharpe={r.sharpe:.4f}, "
            f"DSR={r.dsr:.4f}, PBO={r.pbo:.4f}, Samples={r.samples}"
        )


if __name__ == "__main__":
    main()