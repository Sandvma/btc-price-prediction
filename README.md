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
├—— data/
│   └—— BTC-USD.csv       # Datos históricos de BTC‑USD descargados desde GitHub
├—— images/
│   └—— cumulative_returns.png  # Gráfico generado por el script
├—— model.py             # Script Python para entrenar y evaluar modelos
├—— results.txt          # Métricas de evaluación de los modelos
├—— compare_models.py     # Script que prueba modelos adicionales y guarda métricas en model_results.txt
├—— model_results.txt     # Métricas de los modelos adicionales (regresión y clasificación)
├—— advanced_experiments.py  # Script con horizontes múltiples, indicadores técnicos y validación temporal
├—— advanced_results.txt    # Métricas de los experimentos avanzados
    ├—— enhanced_experiments.py  # Script con retorno logarítmico, purga temporal y métricas robustas
    └—— enhanced_results.txt    # Métricas de los experimentos mejorados
└—— README.md            # Este archivo
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

### Experimentos avanzados

Para abordar algunas de las limitaciones del enfoque inicial, el
repositorio incorpora el script `advanced_experiments.py`. Este módulo
realiza un conjunto más completo de pruebas con las siguientes
características:

* **Horizontes múltiples**: se calculan retornos a 1, 3, 5 y 10 días
  adelante. Esto permite observar cómo evoluciona la calidad de las
  predicciones en ventanas de diferentes longitudes.
* **Indicadores técnicos enriquecidos**: además de los retardos de
  retorno, se incluyen RSI de 14 días, MACD y su línea de señal,
  distancias a las bandas de Bollinger de 20 días y el ATR de 14
  días. Estas variables capturan momentum, tendencia y volatilidad.
* **Métricas adicionales**: se reportan el **Net Points** (suma de los
  cambios de precio en dólares), el **Net Points normalizado por
  ATR** y el **ratio de Sharpe** de una estrategia que compra o vende
  según el signo de la predicción. Estas métricas ayudan a evaluar la
  utilidad práctica de las señales. El cálculo se realiza mediante
  tres particiones temporales para reducir el sesgo de selección.

Para ejecutar estos experimentos y guardar las métricas en
`advanced_results.txt`:

```bash
python advanced_experiments.py
```

Los resultados muestran que, aunque algunos horizontes (por ejemplo,
5 días) mejoran ligeramente la precisión direccional respecto al azar
(≈53 %), el beneficio operativo sigue siendo modesto. Se recomienda
usar estas pruebas como punto de partida para exploraciones más
sofisticadas (por ejemplo, validación *walk‑forward* o modelos
secuenciales) y para analizar el impacto de costes de transacción.

### Experimentos mejorados

Atendiendo a las limitaciones detectadas en las pruebas avanzadas y a
las críticas al uso de métricas poco alineadas con la realidad del
trading, se ha implementado un script adicional,
`enhanced_experiments.py`, que introduce varias mejoras clave:

* **Retorno logarítmico como variable objetivo**: en lugar de
  predecir el cambio de precio en dólares, se modela el retorno
  logarítmico a 1, 3, 5 y 10 días. Esto evita sesgos por la escala del
  activo y se alinea con la forma estándar de medir rendimientos.
* **Validación cruzada con purga temporal**: el conjunto de datos se
  divide en varias particiones temporales. Para cada una, se entrena
  con datos históricos hasta una fecha de corte, se omite una pequeña
  franja de observaciones (embargo) y se evalúa en un bloque posterior.
  Este método reduce la contaminación temporal y se inspira en la
  estrategia de validación de López de Prado.
* **Métricas robustas de trading**: además del MSE y la precisión
  direccional, se calculan el **Net Points** (suma de puntos en
  dólares), el **Net Points normalizado por ATR** (para ajustar la
  volatilidad) y el **ratio de Sharpe** de una estrategia simple que
  toma posiciones según el signo de la predicción. Estas métricas
  permiten juzgar mejor la utilidad real de las señales.

Para ejecutar este experimento mejorado y guardar los resultados en
`enhanced_results.txt` basta con ejecutar:

```bash
python enhanced_experiments.py
```

Los resultados obtenidos (ver `enhanced_results.txt`) muestran que la
precisión direccional apenas supera el 52 % y que la estrategia basada
en el signo de la predicción obtiene ratios de Sharpe modestos (<0,3).
Además, los **Net Points** predichos son negativos en todos los
horizontes, mientras que los reales son positivos para horizontes
superiores (efecto de tendencia). Estas cifras refuerzan la conclusión
de que, con indicadores técnicos básicos y modelos de árboles, es
difícil generar señales rentables para BTC‑USD sin incorporar
información adicional (volumen, indicadores on‑chain, sentimiento,
etc.).

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
