"""
advanced_experiments.py
=======================

Este módulo amplía el conjunto de experimentos del proyecto `btc_price_prediction`
para abordar varios de los puntos de crítica planteados al enfoque original.

Los cambios principales son:

* **Horizontes múltiples**: se evalúan retornos futuros a 1, 3, 5 y 10 barras
  (días) en lugar de centrarse solo en el siguiente día. Esto permite
  estudiar cómo cambia la precisión de la señal con diferentes ventanas de
  predicción.
* **Indicadores técnicos enriquecidos**: además de los diez retardos de
  retornos diarios, se calculan el RSI de 14 días, el MACD con su línea de
  señal, la distancia a las bandas de Bollinger de 20 días y el ATR de 14 días.
  Estas variables ayudan a capturar información de tendencia, momentum y
  volatilidad.
* **Métricas de evaluación ampliadas**: además del error cuadrático medio
  (MSE) y la precisión direccional, se calculan las siguientes métricas:

  - **Net Points** (predicho y real): suma de los cambios de precio en
    dólares obtenidos al multiplicar los retornos predichos/actuales por
    el precio de cierre en la fecha de entrada.
  - **Net Points/ATR** (predicho y real): media del cociente entre los
    cambios de precio y el ATR, para normalizar por la volatilidad.
  - **Sharpe ratio** de la estrategia binaria que compra (o vende) según
    el signo de la predicción y mantiene la posición durante N días. El
    ratio se calcula sin anualizar, multiplicando por la raíz del número
    de observaciones.

* **Validación cronológica con varias particiones**: se implementa un
  esquema simple de validación cruzada temporal con tres particiones
  consecutivas. Cada partición utiliza un tramo inicial como entrenamiento
  (60 %, 70 % y 80 % respectivamente) y un tramo subsecuente del 20 %
  como prueba. Esto evita la fuga de información entre entrenamiento y
  evaluación y proporciona una estimación más robusta de las métricas.

Para ejecutar los experimentos y generar el archivo de resultados, basta
con ejecutar este script desde la raíz del repositorio:

```bash
python advanced_experiments.py
```

El script guardará las métricas obtenidas en `advanced_results.txt`.

"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores técnicos y retardos de retorno al DataFrame.

    Esta función calcula el retorno diario, RSI, MACD, señal del MACD,
    bandas de Bollinger y ATR. También agrega retardos de retorno (lags)
    hasta 10 días.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas 'Adj Close', 'High', 'Low'.

    Returns
    -------
    pd.DataFrame
        El DataFrame original con nuevas columnas de indicadores.
    """
    df = df.copy()
    # Retorno diario
    df['Return1'] = df['Adj Close'].pct_change()
    # RSI 14 días
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    window = 14
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD y señal
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
    # True Range y ATR (14 días)
    high = df['High']
    low = df['Low']
    close_prev = df['Adj Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    window_atr = 14
    df['ATR14'] = true_range.rolling(window_atr).mean()
    # Retardos de retorno
    for i in range(1, 11):
        df[f'Return_lag_{i}'] = df['Return1'].shift(i)
    return df


def evaluate_horizon(data: pd.DataFrame, N: int, n_splits: int = 3):
    """
    Evalúa un modelo ExtraTreesRegressor para un horizonte dado mediante
    validación cruzada temporal.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con las *features* técnicas y la columna 'Adj Close'.
    N : int
        Horizonte de predicción (número de barras hacia adelante).
    n_splits : int
        Número de particiones de validación cruzada.

    Returns
    -------
    dict
        Diccionario con métricas medias a través de las particiones.
    """
    # Generar variable objetivo: retorno de N barras
    df = data.copy()
    df[f'Target_{N}'] = df['Adj Close'].shift(-N) / df['Adj Close'] - 1
    feature_cols = [f'Return_lag_{i}' for i in range(1, 11)] + [
        'RSI14', 'MACD', 'Signal', 'BollingerUpperDist', 'BollingerLowerDist', 'ATR14'
    ]
    # Preparar dataset y eliminar NaN
    # Nota: 'ATR14' se incluye en feature_cols. Evitamos duplicarla en el
    # DataFrame para que dataset['ATR14'] devuelva una serie unidimensional en
    # lugar de un DataFrame con columnas duplicadas.
    dataset = df[feature_cols + [f'Target_{N}', 'Adj Close']].dropna().reset_index(drop=True)
    X = dataset[feature_cols].values
    y = dataset[f'Target_{N}'].values
    prices = dataset['Adj Close'].values
    # ATR14 forma parte de las features, pero la extraemos aquí para usarla
    # como escala en las métricas de Net Points. Convertimos a array 1D.
    atrs = dataset['ATR14'].to_numpy()
    n_samples = len(X)
    # Definir el tamaño de test como 20 % del conjunto
    test_size = int(n_samples * 0.2)
    # Guardar métricas de cada split
    metrics_per_split = []
    for split_idx in range(n_splits):
        # Determinar índices de corte para entrenamiento y prueba
        train_end = int(n_samples * (0.6 + 0.1 * split_idx))
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        price_test = prices[test_start:test_end]
        atr_test = atrs[test_start:test_end]
        # Entrenar modelo
        model = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Métricas
        mse = mean_squared_error(y_test, preds)
        dir_acc = np.mean((preds >= 0) == (y_test >= 0))
        # Net points y Net points/ATR
        pred_delta_price = preds * price_test
        actual_delta_price = y_test * price_test
        net_points_pred = pred_delta_price.sum()
        net_points_actual = actual_delta_price.sum()
        net_points_atr_pred = (pred_delta_price / atr_test).mean()
        net_points_atr_actual = (actual_delta_price / atr_test).mean()
        # Estrategia binaria: signo de la predicción
        strat_returns = np.sign(preds) * y_test
        mean_ret = strat_returns.mean()
        std_ret = strat_returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(len(strat_returns)) if std_ret != 0 else 0
        metrics_per_split.append([
            mse, dir_acc, net_points_pred, net_points_actual,
            net_points_atr_pred, net_points_atr_actual, sharpe
        ])
    # Media de métricas en todas las particiones
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
    """Ejecutar experimentos para diferentes horizontes y guardar resultados."""
    # Cargar datos
    data_path = os.path.join('data', 'BTC-USD.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = df.reset_index(drop=True)
    # Calcular features
    df_feat = compute_technical_features(df)
    horizons = [1, 3, 5, 10]
    results = {}
    for h in horizons:
        results[h] = evaluate_horizon(df_feat, h, n_splits=3)
    # Guardar en archivo
    out_path = 'advanced_results.txt'
    with open(out_path, 'w') as f:
        f.write('Resultados de experimentos avanzados\n')
        f.write('===================================\n')
        f.write('Horizonte\tMSE\tPrecisión\tNetPts_pred\tNetPts_real\tNetPtsATR_pred\tNetPtsATR_real\tSharpe\n')
        for h in horizons:
            r = results[h]
            f.write(f'{h}\t{r["mse"]:.8f}\t{r["dir_acc"]:.4f}\t{r["net_points_pred"]:.2f}\t{r["net_points_actual"]:.2f}\t{r["net_points_atr_pred"]:.4f}\t{r["net_points_atr_actual"]:.4f}\t{r["sharpe"]:.4f}\n')

    print('Experimentos completados. Resultados guardados en', out_path)


if __name__ == '__main__':
    main()