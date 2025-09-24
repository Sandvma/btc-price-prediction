"""
compare_models.py
==================

Este módulo amplía las pruebas del proyecto BTC‑USD para evaluar
múltiples modelos de regresión y clasificación con un conjunto más
amplio de *features* técnicas. El objetivo es explorar si alguno de
ellos mejora sustancialmente la capacidad de predecir la dirección del
retorno de Bitcoin a un día vista.

Modelos evaluados (regresión):

* Regresión lineal
* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor
* Support Vector Regressor (SVR) con kernel RBF
* K‑Nearest Neighbors Regressor

Modelos evaluados (clasificación):

* Regresión logística
* Random Forest Classifier
* Support Vector Classifier (SVC)

Para cada modelo se calculan el error cuadrático medio (MSE) y la
precisión direccional (porcentaje de aciertos en el signo del retorno
predicho). Para los modelos de clasificación se informa únicamente la
precisión. Los resultados se guardan en `model_results.txt` al final
de la ejecución.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_features(df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """
    Crear un DataFrame con *features* técnicos y objetivo desplazado.

    Se incluyen:
    - Lags de retornos diarios.
    - Diferencia de la cotización con respecto a medias móviles de 5 y 10 días.
    - Volatilidad (desviación estándar) de retornos en ventanas de 5 y 10 días.
    - Objetivo: retorno del día siguiente (columna 'Target') para regresión,
      y etiqueta binaria (1 si retorno > 0, 0 en otro caso) para clasificación.
    """
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    # Lags
    for i in range(1, lags + 1):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
    # Medias móviles y diferencias
    df['MA5'] = df['Adj Close'].rolling(5).mean()
    df['MA10'] = df['Adj Close'].rolling(10).mean()
    df['MA_diff_5'] = (df['Adj Close'] - df['MA5']) / df['MA5']
    df['MA_diff_10'] = (df['Adj Close'] - df['MA10']) / df['MA10']
    # Volatilidad
    df['Vol5'] = df['Return'].rolling(5).std()
    df['Vol10'] = df['Return'].rolling(10).std()
    # Objetivo regresión: retorno del día siguiente
    df['Target'] = df['Return'].shift(-1)
    # Objetivo clasificación
    df['TargetCls'] = (df['Target'] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def evaluate_regression_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values

    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=300, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.001)),
        ]),
        'KNN': KNeighborsRegressor(n_neighbors=5),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        acc = np.mean((preds >= 0) == (y_test >= 0))
        results.append((name, mse, acc))
    return results


def evaluate_classification_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    X_train = train_df[feature_cols].values
    y_train = train_df['TargetCls'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['TargetCls'].values

    # Escalado para modelos que lo requieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cls_models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=300, random_state=42),
        'SVC': SVC(kernel='rbf', C=1.0, gamma='scale'),
    }
    cls_results = []
    for name, model in cls_models.items():
        if name == 'RandomForestClassifier':
            # Random forest no necesita escalado
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        cls_results.append((name, acc))
    return cls_results


def main():
    data_path = os.path.join('data', 'BTC-USD.csv')
    df = load_data(data_path)
    df_feat = build_features(df, lags=10)
    feature_cols = [col for col in df_feat.columns if col.startswith('Return_lag_')] + [
        'MA_diff_5', 'MA_diff_10', 'Vol5', 'Vol10'
    ]
    train_df, test_df = split_train_test(df_feat, test_size=0.2)
    reg_results = evaluate_regression_models(train_df, test_df, feature_cols)
    cls_results = evaluate_classification_models(train_df, test_df, feature_cols)

    # Escribir resultados a archivo
    with open('model_results.txt', 'w') as f:
        f.write('Resultados de modelos de regresión\n')
        f.write('=================================\n')
        f.write('Modelo\tMSE\tPrecisión direccional\n')
        for name, mse, acc in reg_results:
            f.write(f'{name}\t{mse:.8f}\t{acc:.4f}\n')
        f.write('\n')
        f.write('Resultados de modelos de clasificación\n')
        f.write('=====================================\n')
        f.write('Modelo\tPrecisión\n')
        for name, acc in cls_results:
            f.write(f'{name}\t{acc:.4f}\n')


if __name__ == '__main__':
    main()
