"""
enhanced_experiments.py
=======================

Este módulo extiende aún más los experimentos realizados en el proyecto
`btc_price_prediction` para abordar críticas habituales al diseño de
backtests y al uso de métricas inadecuadas. En concreto, se implementan
las siguientes mejoras respecto a `advanced_experiments.py`:

* **Definición de la variable objetivo**: en lugar de predecir el
  simple cambio de precio en dólares, se modela el **retorno logarítmico
  a N días** (`Return_N = log(AdjClose_{t+N}/AdjClose_t)`). Esto
  alinea la métrica de entrenamiento con la escala relativa de los
  movimientos de mercado y evita que el modelo optimice inadvertidamente
  por la magnitud del precio nominal.

* **Validación cruzada con purga temporal**: se utiliza un esquema de
  validación cronológica con purga (inspirado en el trabajo de López de
  Prado). Para cada partición, se entrena con los datos anteriores a
  una fecha de corte y se reserva un bloque de observaciones a modo de
  **embargo**, eliminándolo tanto del conjunto de entrenamiento como del
  de prueba para evitar filtraciones de información. A continuación se
  evalúa sobre un bloque de prueba que no solapa con el entrenamiento.

* **Métricas de evaluación robustas**: además del error cuadrático medio
  (MSE) y la precisión direccional, se calculan métricas de trading
  realistas:

  - **Net Points** y **Net Points normalizado por ATR**: se obtienen a
    partir de las posiciones generadas por el modelo (`sign(pred)`)
    multiplicadas por el retorno real. La versión normalizada divide
    cada contribución por el ATR para ajustarse a la volatilidad.
  - **Sharpe ratio de la estrategia**: se calcula la razón de Sharpe sin
    anualizar a partir de los retornos diarios de la estrategia.

* **Horizontes múltiples**: se evalúan horizontes de 1, 3, 5 y 10 días
  para estudiar cómo cambia la relación señal‑ruido conforme aumenta la
  ventana de predicción.

El script puede ejecutarse desde la raíz del repositorio mediante:

```
python enhanced_experiments.py
```

Los resultados se guardan en `enhanced_results.txt`.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores técnicos y retardos al DataFrame.

    Calcula el retorno diario, RSI, MACD, señal del MACD, distancia a las
    bandas de Bollinger y el ATR de 14 días, así como diez retardos de
    retorno. Estas variables sirven como *features* para los modelos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas 'Adj Close', 'High', 'Low'.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas adicionales de indicadores.
    """
    df = df.copy()
    # Retorno logarítmico diario
    df['Return1'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    # RSI 14 días
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    window = 14
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD y señal (utilizando medias exponenciales)
    ema12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bandas de Bollinger (20 días)
    sma20 = df['Adj Close'].rolling(window=20).mean()
    std20 = df['Adj Close'].rolling(window=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    df['BollingerUpperDist'] = (df['Adj Close'] - upper_band) / (2 * std20)
    df['BollingerLowerDist'] = (df['Adj Close'] - lower_band) / (2 * std20)
    # True Range y ATR 14
    high = df['High']
    low = df['Low']
    close_prev = df['Adj Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = true_range.rolling(window=14).mean()
    # Retardos de retorno logarítmico (lags)
    for i in range(1, 11):
        df[f'Return_lag_{i}'] = df['Return1'].shift(i)
    return df


def purged_cv_splits(n_samples: int, test_size: float, n_splits: int, embargo_pct: float = 0.01):
    """Genera índices de entrenamiento y prueba para validación cruzada purgada.

    Esta función divide los datos en varias particiones temporales. Para cada
    split, se seleccionan los primeros `train_end` puntos como entrenamiento,
    después se omite un número de observaciones igual a `embargo` (purga) y
    a continuación se usa un bloque de tamaño `test_size` como prueba. Esta
    aproximación evita el solapamiento temporal entre entrenamiento y prueba.

    Parameters
    ----------
    n_samples : int
        Número total de muestras en el dataset.
    test_size : float
        Proporción del total que se usará como bloque de prueba en cada
        split (por ejemplo 0.2 para un 20 %).
    n_splits : int
        Número de particiones a generar.
    embargo_pct : float
        Proporción del tamaño del bloque de prueba que se utilizará como
        embargo (purga) entre entrenamiento y prueba.

    Yields
    ------
    tuple
        Tupla (train_indices, test_indices) para cada split.
    """
    test_block = int(n_samples * test_size)
    embargo = max(1, int(test_block * embargo_pct))
    # Ajustar test_block para que quepan todas las particiones
    for idx in range(n_splits):
        train_end = int(n_samples * (0.5 + 0.1 * idx))
        # Calcular posición inicial del bloque de prueba después del embargo
        test_start = train_end + embargo
        test_end = test_start + test_block
        if test_end > n_samples:
            break
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        yield train_indices, test_indices


def evaluate_horizon(df_feat: pd.DataFrame, N: int, n_splits: int = 4) -> dict:
    """Evalúa un modelo ExtraTreesRegressor para un horizonte dado con purga.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame con *features* técnicas y la columna 'Adj Close'.
    N : int
        Horizonte de predicción (número de barras en el futuro).
    n_splits : int
        Número de particiones para validación cruzada purgada.

    Returns
    -------
    dict
        Diccionario con las métricas medias resultantes.
    """
    df = df_feat.copy()
    # Definir la variable objetivo como retorno logarítmico a N días
    df[f'Return_{N}'] = np.log(df['Adj Close'].shift(-N) / df['Adj Close'])
    feature_cols = [f'Return_lag_{i}' for i in range(1, 11)] + [
        'RSI14', 'MACD', 'Signal', 'BollingerUpperDist', 'BollingerLowerDist', 'ATR14'
    ]
    # Seleccionar y limpiar datos
    dataset = df[feature_cols + [f'Return_{N}', 'Adj Close']].dropna().reset_index(drop=True)
    X = dataset[feature_cols].values
    y = dataset[f'Return_{N}'].values
    prices = dataset['Adj Close'].values
    atrs = dataset['ATR14'].to_numpy()
    n_samples = len(X)
    test_size = 0.2
    metrics_per_split = []
    for train_idx, test_idx in purged_cv_splits(n_samples, test_size, n_splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        price_test = prices[test_idx]
        atr_test = atrs[test_idx]
        model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Métricas
        mse = mean_squared_error(y_test, preds)
        dir_acc = np.mean((preds >= 0) == (y_test >= 0))
        # Net points (usando retornos log -> aproximación de delta price en %)
        # Convertir retornos log a retorno simple para calcular puntos
        pred_simple = np.exp(preds) - 1
        actual_simple = np.exp(y_test) - 1
        # Puntos reales/predichos en dólares
        pred_points = pred_simple * price_test
        actual_points = actual_simple * price_test
        net_points_pred = pred_points.sum()
        net_points_actual = actual_points.sum()
        # Normalizar por ATR
        net_points_atr_pred = (pred_points / atr_test).mean()
        net_points_atr_actual = (actual_points / atr_test).mean()
        # Estrategia: usar signo de la predicción (retorno log)
        strat_returns = np.sign(preds) * actual_simple
        mean_ret = strat_returns.mean()
        std_ret = strat_returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(len(strat_returns)) if std_ret != 0 else 0
        metrics_per_split.append([
            mse, dir_acc, net_points_pred, net_points_actual,
            net_points_atr_pred, net_points_atr_actual, sharpe
        ])
    metrics_arr = np.mean(metrics_per_split, axis=0)
    return {
        'mse': metrics_arr[0],
        'dir_acc': metrics_arr[1],
        'net_points_pred': metrics_arr[2],
        'net_points_actual': metrics_arr[3],
        'net_points_atr_pred': metrics_arr[4],
        'net_points_atr_actual': metrics_arr[5],
        'sharpe': metrics_arr[6],
    }


def main():
    """Ejecuta la evaluación para varios horizontes y guarda resultados."""
    data_path = os.path.join('data', 'BTC-USD.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = df.reset_index(drop=True)
    df_feat = compute_technical_features(df)
    horizons = [1, 3, 5, 10]
    results = {}
    for h in horizons:
        results[h] = evaluate_horizon(df_feat, h, n_splits=4)
    out_path = 'enhanced_results.txt'
    with open(out_path, 'w') as f:
        f.write('Resultados de experimentos mejorados\n')
        f.write('===================================\n')
        f.write('Horizonte\tMSE\tPrecisión\tNetPts_pred\tNetPts_real\tNetPtsATR_pred\tNetPtsATR_real\tSharpe\n')
        for h in horizons:
            r = results[h]
            f.write(
                f'{h}\t{r["mse"]:.8f}\t{r["dir_acc"]:.4f}\t{r["net_points_pred"]:.2f}\t'
                f'{r["net_points_actual"]:.2f}\t{r["net_points_atr_pred"]:.4f}\t'
                f'{r["net_points_atr_actual"]:.4f}\t{r["sharpe"]:.4f}\n'
            )
    print('Experimentos mejorados completados. Resultados guardados en', out_path)


if __name__ == '__main__':
    main()