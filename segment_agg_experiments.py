"""
segment_agg_experiments.py
===========================

Este módulo explora una aproximación inspirada en la literatura de
``Transformers`` con atención y segmentación adaptativa, pero con
herramientas disponibles en este entorno (sin TensorFlow/PyTorch). La
idea principal es segmentar la serie temporal en subseries dinámicas
basadas en movimientos relativos del precio del ``BTC-USD`` y construir
características agregadas por segmento. Cada subserie se resume por
estadísticas como la duración, el retorno acumulado, la volatilidad y
los promedios de indicadores técnicos. A continuación se crea un
conjunto de datos a nivel de segmento en el que la variable objetivo
indica si el retorno de la siguiente subserie es positivo o no. Este
enfoque emula la noción de modelar patrones de comportamiento
similares por separado, presente en los ``Temporal Fusion Transformers``
con segmentación adaptativa【699532840740462†L654-L713】.

El script realiza los siguientes pasos:

* Calcula indicadores técnicos sobre los precios diarios: retorno
  logarítmico, RSI de 14 días, MACD y su señal, distancias a las
  bandas de Bollinger y ATR14, además de otras derivadas de la serie.
* Segmenta la serie de precios ajustados usando un umbral relativo
  (por defecto 5 %) para detectar máximos locales y separar subseries.
* Para cada segmento calcula un vector de características agregadas,
  incluyendo la longitud, el retorno acumulado, la volatilidad,
  estadísticas de los indicadores técnicos y una pendiente lineal
  (slope) de la tendencia interna.
* Asigna como etiqueta ``1`` si el retorno del siguiente segmento es
  positivo y ``0`` en caso contrario; se descartan los últimos
  segmentos sin sucesor.
* Entrena un clasificador ``ExtraTreesClassifier`` para predecir la
  etiqueta usando validación temporal (80 % para entrenamiento,
  20 % para prueba) y calcula métricas como precisión y ratio de
  Sharpe de una estrategia de trading que sigue la señal del modelo.

Este experimento pretende capturar patrones de varias escalas y
dinámicas sin depender de librerías de deep learning, mostrando cómo
una segmentación adaptativa puede enriquecer las características y
potencialmente mejorar la previsión de dirección.

Para ejecutar el script desde la raíz del repositorio:

```
python segment_agg_experiments.py
```

Los resultados se guardan en ``segment_agg_results.txt`` en el
directorio raíz del proyecto.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos básicos para la serie de precios.

    Se calculan retornos logarítmicos, RSI14, MACD y su señal,
    distancias a las bandas de Bollinger y ATR14. Devuelve un DataFrame
    con columnas adicionales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas ``Adj Close``, ``High`` y ``Low``.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas de indicadores técnicos.
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
    rs = avg_gain / (avg_loss + 1e-12)
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
    df['BollingerUpperDist'] = (df['Adj Close'] - upper_band) / (2 * std20 + 1e-12)
    df['BollingerLowerDist'] = (df['Adj Close'] - lower_band) / (2 * std20 + 1e-12)
    # ATR 14
    high = df['High']
    low = df['Low']
    close_prev = df['Adj Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = true_range.rolling(window=14).mean()
    return df


def segment_series(prices: np.ndarray, threshold: float) -> list:
    """Segmenta una serie de precios en subseries basadas en aumentos relativos.

    Un nuevo segmento comienza cada vez que el precio actual crece más
    del ``threshold`` respecto al mínimo local observado en el segmento
    actual. Devuelve una lista de tuplas (inicio, fin) con índices de
    cada segmento.

    Parameters
    ----------
    prices : np.ndarray
        Array de precios (por ejemplo, `Adj Close`).
    threshold : float
        Umbral relativo (por ejemplo 0.05 para 5 %) para cerrar un segmento.

    Returns
    -------
    list
        Lista de pares (start, end) que delimitan cada subserie.
    """
    n = len(prices)
    segments = []
    current_min = prices[0]
    start_idx = 0
    for i in range(1, n):
        # Actualizar mínimo
        if prices[i] < current_min:
            current_min = prices[i]
        # Cerrar segmento si se supera el umbral
        if (prices[i] - current_min) / (current_min + 1e-12) >= threshold:
            segments.append((start_idx, i))
            start_idx = i + 1
            if start_idx < n:
                current_min = prices[start_idx]
    # Añadir último segmento (si hay espacio)
    if start_idx < n - 1:
        segments.append((start_idx, n - 1))
    return segments


def compute_segment_features(df_feat: pd.DataFrame, segments: list) -> pd.DataFrame:
    """Crea un DataFrame de características agregadas por segmento.

    Cada fila corresponde a un segmento y contiene estadísticas
    resumidas: longitud, retorno acumulado, volatilidad, media y
    desviación típica de los retornos, promedios de los indicadores
    técnicos y la pendiente de una regresión lineal sobre el log-precio.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame original con precios e indicadores técnicos.
    segments : list
        Lista de tuplas (start, end) con índices de cada segmento.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por segmento y columnas de features.
    """
    rows = []
    for (s, e) in segments:
        # segment slice
        seg = df_feat.iloc[s:e+1]
        if seg.shape[0] < 2:
            continue
        length = seg.shape[0]
        # Retorno acumulado (log) y simple
        ret_simple = (seg['Adj Close'].iloc[-1] / seg['Adj Close'].iloc[0]) - 1
        ret_log = np.log(seg['Adj Close'].iloc[-1] / seg['Adj Close'].iloc[0])
        # Desviación estándar de retornos diarios
        vol = seg['Return1'].std()
        mean_ret = seg['Return1'].mean()
        # Promedios de indicadores
        rsi_mean = seg['RSI14'].mean()
        macd_mean = seg['MACD'].mean()
        signal_mean = seg['Signal'].mean()
        boll_upper_mean = seg['BollingerUpperDist'].mean()
        boll_lower_mean = seg['BollingerLowerDist'].mean()
        atr_mean = seg['ATR14'].mean()
        # Slope de regresión lineal de log precio vs índice
        y = np.log(seg['Adj Close'].values)
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
        rows.append({
            'start': s,
            'end': e,
            'length': length,
            'ret_simple': ret_simple,
            'ret_log': ret_log,
            'vol': vol,
            'mean_ret': mean_ret,
            'rsi_mean': rsi_mean,
            'macd_mean': macd_mean,
            'signal_mean': signal_mean,
            'boll_upper_mean': boll_upper_mean,
            'boll_lower_mean': boll_lower_mean,
            'atr_mean': atr_mean,
            'slope': slope
        })
    seg_df = pd.DataFrame(rows)
    return seg_df


def prepare_dataset(df_feat: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Prepara el conjunto de datos a nivel de segmento.

    Segmenta la serie y calcula las features agregadas; luego asigna
    la etiqueta como 1 si el retorno logarítmico del siguiente
    segmento es positivo y 0 en caso contrario. El último segmento
    que no tiene sucesor se descarta.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame con precios e indicadores técnicos.
    threshold : float
        Umbral para la segmentación.

    Returns
    -------
    pd.DataFrame
        DataFrame con features agregadas por segmento y la etiqueta.
    """
    prices = df_feat['Adj Close'].values
    segments = segment_series(prices, threshold)
    seg_df = compute_segment_features(df_feat, segments)
    # Calcular etiqueta: retorno del siguiente segmento
    labels = []
    for i in range(len(seg_df) - 1):
        next_ret = seg_df.loc[i + 1, 'ret_log']
        labels.append(1 if next_ret > 0 else 0)
    seg_df = seg_df.iloc[:-1].copy()
    seg_df['label'] = labels
    # Eliminar columnas start y end que no son útiles para el modelo
    seg_df = seg_df.drop(columns=['start', 'end'])
    seg_df = seg_df.dropna()
    return seg_df


def train_and_evaluate(seg_df: pd.DataFrame) -> dict:
    """Entrena un modelo ExtraTreesClassifier y evalúa precisión y Sharpe.

    Utiliza una partición temporal 80/20 para entrenamiento y prueba.
    La estrategia de trading toma posición larga si la probabilidad de
    clase 1 supera 0.5 y corta en caso contrario. Se calcula la
    precisión de clasificación y el ratio de Sharpe del PnL de la
    estrategia.

    Parameters
    ----------
    seg_df : pd.DataFrame
        Conjunto de datos con features y etiqueta.

    Returns
    -------
    dict
        Diccionario con métricas de precisión y Sharpe.
    """
    # Orden cronológico implícito
    n = len(seg_df)
    split = int(n * 0.8)
    train = seg_df.iloc[:split]
    test = seg_df.iloc[split:]
    X_train = train.drop(columns=['label'])
    y_train = train['label']
    X_test = test.drop(columns=['label'])
    y_test = test['label']
    # Entrenar clasificador
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    # Predicciones probabilísticas
    probas = clf.predict_proba(X_test)[:, 1]
    preds = (probas >= 0.5).astype(int)
    # Precisión
    accuracy = accuracy_score(y_test, preds)
    # PnL de la estrategia: retorno logarítmico del siguiente segmento
    # (shifted) multiplicado por signo de predicción (1: largo, 0: corto)
    # Para un corto interpretamos -retorno (tomar signo invertido)
    # Convertimos a +1/-1
    signals = np.where(preds == 1, 1, -1)
    # Retornos reales del período de prueba (próximo segmento)
    returns_test = seg_df.iloc[split:]['ret_log'].values[1:]
    # Ajustar longitud para que coincida con señales (un elemento menos)
    signals = signals[:-1]
    # PnL
    pnl = signals * returns_test
    if pnl.size > 1 and np.nanstd(pnl) > 0:
        sharpe = np.nanmean(pnl) / (np.nanstd(pnl) + 1e-12) * np.sqrt(len(pnl))
    else:
        sharpe = 0.0
    return {'accuracy': accuracy, 'sharpe': sharpe}


def main():
    # Cargar datos
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'BTC-USD.csv')
    df = pd.read_csv(data_path)
    df_feat = compute_technical_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    # Usar un umbral del 5 % para segmentación
    threshold = 0.05
    seg_dataset = prepare_dataset(df_feat, threshold)
    if seg_dataset.empty or seg_dataset.shape[0] < 30:
        print("El conjunto de datos de segmentos es demasiado pequeño para entrenar un modelo.")
        return
    results = train_and_evaluate(seg_dataset)
    # Guardar resultados
    out_path = os.path.join(os.path.dirname(__file__), 'segment_agg_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"Modelo segmentado con umbral {threshold:.2f}\n")
        f.write(f"Número de segmentos: {len(seg_dataset)}\n")
        f.write(f"Precisión: {results['accuracy']:.4f}\n")
        f.write(f"Sharpe: {results['sharpe']:.4f}\n")
    print(f"Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()