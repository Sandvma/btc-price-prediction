# BTC‑USD Price Prediction Tests

Este proyecto implementa un conjunto de pruebas simples de predicción de precios
usando datos históricos de la cotización BTC‑USD de Yahoo Finance. La idea es
construir un conjunto de *features* basadas en retornos pasados y evaluar
modelos básicos de regresión para predecir el retorno diario. El objetivo es
comparar la capacidad de un modelo lineal sencillo frente a un modelo no lineal
como un **Random Forest** para capturar patrones en los datos. No se incluye
ningún modelo de redes neuronales profundas debido a la ausencia de dependencias
como TensorFlow o PyTorch en este entorno.

## Estructura del proyecto

```
btc_price_prediction/
├── data/
│   └── BTC-USD.csv       # Datos históricos de BTC‑USD descargados desde GitHub
├── images/
│   └── cumulative_returns.png  # Gráfico generado por el script
├── model.py             # Script Python para entrenar y evaluar modelos
├── results.txt          # Métricas de evaluación de los modelos
├── compare_models.py     # Script que prueba modelos adicionales y guarda métricas en model_results.txt
├── model_results.txt     # Métricas de los modelos adicionales (regresión y clasificación)
└── README.md            # Este archivo
```

## Dependencias

El script `model.py` utiliza únicamente bibliotecas disponibles en la
distribución estándar de este entorno:

* **pandas** para manipulación de datos.
* **numpy** para operaciones numéricas.
* **scikit‑learn** (`sklearn`) para modelos de regresión.
* **matplotlib** para visualización.

No es necesario instalar paquetes adicionales.

## Ejecución

Para reproducir las pruebas, basta con ejecutar el script `model.py` desde la
raíz del repositorio:

```bash
python model.py
```

El script realizará los siguientes pasos:

1. Carga y ordena los datos históricos desde `data/BTC-USD.csv`.
2. Calcula el retorno diario basado en el precio de cierre ajustado.
3. Construye *features* usando los diez retornos diarios anteriores (lags).
4. Divide cronológicamente los datos en un 80 % para entrenamiento y un 20 %
   para prueba.
5. Entrena dos modelos:
   - **Regresión lineal** (como línea base).
   - **Random Forest Regressor** con 200 árboles.
6. Evalúa cada modelo usando el **error cuadrático medio (MSE)** y la
   **precisión direccional** (porcentaje de veces que el signo del retorno
   predicho coincide con el signo real).
7. Guarda las métricas en `results.txt` y genera un gráfico comparando
   el retorno acumulado real con el retorno acumulado predicho por el
   Random Forest en el conjunto de prueba.

Además del script principal, puedes ejecutar `compare_models.py` para
evaluar otros modelos de regresión (Extra Trees, Gradient Boosting, SVR,
KNN) y modelos de clasificación (regresión logística, Random Forest
Classifier, Support Vector Classifier). Este script guarda los
resultados en `model_results.txt`. Los experimentos adicionales muestran
que, con las *features* simples utilizadas y los datos diarios de
BTC‑USD, ninguna de las técnicas probadas supera de forma significativa
el 50 % de precisión direccional, aunque el modelo SVR presenta una
ligera mejora marginal (≈50,3 %).

## Resultados

Tras ejecutar el script se generará un archivo `results.txt` con las métricas
obtenidas y un gráfico `cumulative_returns.png` en el directorio `images/`. Los
resultados pueden interpretarse como un punto de partida para evaluar si las
señales derivadas de retornos pasados aportan valor para predecir el movimiento
futuro del precio. Dado que no se exploran hiperparámetros ni modelos
complejos, se espera que la capacidad predictiva sea limitada.

## Consideraciones finales

* Esta prueba utiliza datos históricos estáticos hasta julio de 2023. Para
  replicarla con datos más recientes, basta con reemplazar el archivo
  `BTC-USD.csv` en la carpeta `data` por una versión actualizada.
* Se recomienda extender este experimento incorporando técnicas de
  validación *walk‑forward* o *rolling* para evitar sobreajuste y explorar
  otros modelos (por ejemplo, redes neuronales recurrentes) en un entorno
  adecuado.
