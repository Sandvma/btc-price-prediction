"""
segmented_experiments.py
========================

Este módulo introduce un enfoque inspirado en la literatura reciente sobre
transformers con segmentación adaptativa para predecir retornos del par
``BTC-USD``. En lugar de entrenar un único modelo sobre toda la serie, se
divide dinámicamente la serie temporal en segmentos basados en aumentos
relativos del precio y se entrena un modelo independiente para cada
categoría de segmento. Esta idea emula las propuestas de
``Temporal Fusion Transformers`` con segmentación adaptativa, donde cada
subserie de comportamiento similar se modela por separado【699532840740462†L654-L713】.

En concreto, se implementan los siguientes pasos:

* **Cálculo de indicadores técnicos**: igual que en los experimentos
  anteriores, se calculan retornos logarítmicos diarios, RSI de 14 días,
  MACD y su señal, distancias a las bandas de Bollinger y el ATR de 14
  días, así como diez retardos del retorno. Estas variables sirven de
  *features* de entrada.

* **Segmentación dinámica**: se recorre la serie de precios ajustados y
  se detectan incrementos relativos respecto al mínimo local. Cuando el
  precio actual supera el mínimo anterior en más de un umbral
  (por defecto 5 %), se cierra el segmento y se inicia uno nuevo.
  Cada segmento recibe una etiqueta binaria: ``1`` si el precio al final del
  segmento es mayor que el precio al inicio (tendencia alcista) o ``0``
  en caso contrario (tendencia bajista). Este procedimiento está
  inspirado en las subseries que terminan en máximos relativos utilizadas
  en los modelos adaptativos【699532840740462†L654-L713】.

* **Modelos por categoría**: para cada categoría (alcista/bajista) se
  entrena un ``ExtraTreesRegressor`` usando los primeros
  ``80 %`` de muestras de la categoría y se evalúa sobre el ``20 %`` final.
  Se evalúan las siguientes métricas:

  - ``MSE`` (error cuadrático medio) de los retornos logarítmicos.
  - ``Precisión direccional``: porcentaje de veces que el signo de la
    predicción coincide con el signo real.
  - ``Net Points``: suma de los retornos simples de la estrategia que toma
    posición larga si la predicción es positiva y corta si es negativa.
  - ``Net Points por ATR``: ``Net Points`` normalizado dividiendo cada
    contribución por el ``ATR14`` correspondiente, lo que ajusta por
    volatilidad.
  - ``Sharpe``: ratio de Sharpe sin anualizar de la estrategia de
    signado.

* **Exploración de parámetros**: se prueban distintos umbrales de
  segmentación (``3 %``, ``5 %`` y ``10 %``) y horizontes de predicción
  (``1``, ``3``, ``5`` y ``10`` días). Los resultados se guardan en
  ``segmented_results.txt``.

Para ejecutar el script desde la raíz del repositorio:

```
python segmented_experiments.py
```

La metodología y el código buscan ilustrar cómo la segmentación por
patrones de tendencia puede mejorar la señal predicha, de acuerdo con
los artículos que muestran mejoras al modelar subseries de distinta
dinámica【699532840740462†L786-L894】【262867392780735†L1220-L1289】.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos y retardos.

    Copiado de ``enhanced_experiments.py``. Devuelve un DataFrame con
    columnas de retornos logarítmicos, RSI, MACD, señal, distancias a
    bandas de Bollinger, ATR y diez retardos del retorno.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas ``Adj Close``, ``High`` y ``Low``.

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
    # MACD y señal (medias exponenciales)
    ema12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bandas de Bollinger 20 días
    sma20 = df['Adj Close'].rolling(window=20).mean()
    std20 = df['Adj Close'].rolling(window=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    df['BollingerUpperDist'] = (df['Adj Close'] - upper_band) / (2 * std20)
    df['BollingerLowerDist'] = (df['Adj Close'] - lower_band) / (2 * std20)
    # ATR 14
    high = df['High']
    low = df['Low']
    close_prev = df['Adj Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = true_range.rolling(window=14).mean()
    # Lags de retorno logarítmico
    for i in range(1, 11):
        df[f'Return_lag_{i}'] = df['Return1'].shift(i)
    return df


def segment_series(prices: np.ndarray, threshold: float) -> tuple:
    """Segmenta una serie de precios en subseries basadas en aumentos relativos.

    Un nuevo segmento comienza después de que el precio actual haya crecido
    respecto al mínimo observado en el segmento en más de ``threshold``.
    Cada segmento se etiqueta como 1 si el precio final supera al precio
    inicial, y 0 en caso contrario.

    Parameters
    ----------
    prices : np.ndarray
        Array de precios ajustados.
    threshold : float
        Umbral relativo (por ejemplo 0.05 para 5 %) para cerrar un segmento.

    Returns
    -------
    tuple
        Tupla (segment_id, segment_cat) de arrays del mismo tamaño que
        ``prices``, donde ``segment_id[i]`` indica a qué segmento
        pertenece cada observación y ``segment_cat[i]`` la etiqueta del
        segmento (0: bajista, 1: alcista).
    """
    n = len(prices)
    segment_id = np.full(n, -1, dtype=int)
    segment_cat = np.zeros(n, dtype=int)
    current_min = prices[0]
    start_idx = 0
    segments = []
    for i in range(1, n):
        # Actualizar mínimo
        if prices[i] < current_min:
            current_min = prices[i]
        # Cerrar segmento si el precio supera el mínimo en ``threshold``
        if (prices[i] - current_min) / (current_min + 1e-12) >= threshold:
            segments.append((start_idx, i))
            start_idx = i + 1
            if start_idx < n:
                current_min = prices[start_idx]
    # Añadir el último segmento
    if start_idx < n - 1:
        segments.append((start_idx, n - 1))
    # Rellenar los arrays
    for seg_idx, (s, e) in enumerate(segments):
        segment_id[s : e + 1] = seg_idx
        segment_cat[s : e + 1] = int(prices[e] > prices[s])
    return segment_id, segment_cat


def evaluate_segmented_model(df_feat: pd.DataFrame, horizon: int, threshold: float) -> dict:
    """Entrena y evalúa modelos separados por segmento.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame con indicadores técnicos y precios.
    horizon : int
        Horizonte de predicción (número de días en el futuro).
    threshold : float
        Umbral para la segmentación relativa.

    Returns
    -------
    dict
        Diccionario con las métricas de cada categoría y agregadas.
    """
    df = df_feat.copy()
    # Variable objetivo: retorno logarítmico a ``horizon`` días
    df[f'Return_{horizon}'] = np.log(df['Adj Close'].shift(-horizon) / df['Adj Close'])
    # Realizar segmentación
    seg_id, seg_cat = segment_series(df['Adj Close'].to_numpy(), threshold)
    df['segment_cat'] = seg_cat
    # Features
    feat_cols = [f'Return_lag_{i}' for i in range(1, 11)] + [
        'RSI14', 'MACD', 'Signal', 'BollingerUpperDist', 'BollingerLowerDist', 'ATR14'
    ]
    # Filtrar filas válidas
    mask = df[feat_cols + [f'Return_{horizon}']].notnull().all(axis=1)
    # Construir DataFrame válido. Incluimos una sola columna ATR14 para evitar duplicados,
    # ya que ATR14 también forma parte de feat_cols. Si se incluye dos veces,
    # pandas creará columnas duplicadas y ``atr_test`` será de dimensión (n,2).
    cols = feat_cols + [f'Return_{horizon}', 'segment_cat']
    df_valid = df.loc[mask, cols].copy()
    df_valid['ATR14'] = df.loc[mask, 'ATR14'].values
    results = {}
    all_preds = []
    all_actual = []
    all_signs = []
    all_atr = []
    for cat in sorted(df_valid['segment_cat'].unique()):
        sub = df_valid[df_valid['segment_cat'] == cat].copy()
        if len(sub) < 60:
            continue
        X = sub[feat_cols].values
        y = sub[f'Return_{horizon}'].values
        atr = sub['ATR14'].values
        # División cronológica 80/20
        split = int(len(sub) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        atr_test = atr[split:]
        model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        dir_acc = np.mean((preds >= 0) == (y_test >= 0))
        # Calcular métricas de trading
        actual_simple = np.exp(y_test) - 1
        signal = np.sign(preds)
        strategy_returns = signal * actual_simple
        net_points = strategy_returns.sum()
        net_points_atr = (signal * actual_simple / atr_test).sum()
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8)
        results[cat] = {
            'n': len(y_test),
            'mse': mse,
            'dir_acc': dir_acc,
            'net_pts': net_points,
            'net_pts_atr': net_points_atr,
            'sharpe': sharpe,
        }
        # Agregar a agregados
        all_preds.append(preds)
        all_actual.append(y_test)
        all_signs.append(signal)
        all_atr.append(atr_test)
    # Agregados
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_actual = np.concatenate(all_actual)
        all_signs = np.concatenate(all_signs)
        all_atr = np.concatenate(all_atr)
        w_mse = sum(results[c]['mse'] * results[c]['n'] for c in results) / sum(
            results[c]['n'] for c in results
        )
        w_dir = sum(results[c]['dir_acc'] * results[c]['n'] for c in results) / sum(
            results[c]['n'] for c in results
        )
        w_net_pts = sum(results[c]['net_pts'] for c in results)
        w_net_pts_atr = sum(results[c]['net_pts_atr'] for c in results)
        # Sharpe global
        overall_strategy = all_signs * (np.exp(all_actual) - 1)
        w_sharpe = overall_strategy.mean() / (overall_strategy.std() + 1e-8)
        results['aggregate'] = {
            'n': sum(results[c]['n'] for c in results),
            'mse': w_mse,
            'dir_acc': w_dir,
            'net_pts': w_net_pts,
            'net_pts_atr': w_net_pts_atr,
            'sharpe': w_sharpe,
        }
    return results


def main() -> None:
    """Punto de entrada del script.

    Carga los datos de ``BTC-USD`` desde ``data/BTC-USD.csv``, calcula
    indicadores técnicos, explora diferentes umbrales de segmentación y
    horizontes de predicción, evalúa los modelos segmentados y guarda
    los resultados en ``segmented_results.txt``.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'BTC-USD.csv')
    df = pd.read_csv(data_path)
    # Calcular indicadores
    df_feat = compute_technical_features(df)
    thresholds = [0.03, 0.05, 0.10]
    horizons = [1, 3, 5, 10]
    lines = []
    lines.append('Threshold,Horizon,Category,NSamples,MSE,DirAcc,NetPts,NetPtsATR,Sharpe')
    for thr in thresholds:
        for h in horizons:
            res = evaluate_segmented_model(df_feat, h, thr)
            for cat, metrics in res.items():
                lines.append(
                    f"{thr},{h},{cat},{metrics['n']},{metrics['mse']:.6f},{metrics['dir_acc']:.4f},"
                    f"{metrics['net_pts']:.4f},{metrics['net_pts_atr']:.4f},{metrics['sharpe']:.4f}"
                )
    # Guardar resultados
    out_path = os.path.join(base_dir, 'segmented_results.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Resultados guardados en {out_path}")


if __name__ == '__main__':
    main()