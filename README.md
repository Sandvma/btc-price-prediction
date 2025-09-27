# Proyecto de predicción y pruebas cuantitativas con BTC‑USD

Este repositorio recopila una serie de **experimentos de trading algorítmico** basados en el par BTC‑USD.  A lo largo del desarrollo se han incorporado diferentes enfoques: modelos clásicos de regresión y clasificación, algoritmos de *gradient boosting*, segmentación adaptativa de series, aprendizaje por refuerzo y finalmente un pipeline completo inspirado en la *Fábrica de Estrategias Cuantitativas*.  Cada módulo se mantiene aquí para servir como referencia didáctica y comparativa; los ficheros obsoletos o redundantes han sido eliminados.

## Datos

La carpeta `data/` contiene los datasets utilizados:

| Archivo                          | Descripción |
|---------------------------------|-------------|
| `BTC-USD.csv`                   | Serie diaria OHLCV de BTC‑USD con cierres ajustados procedente de Yahoo Finance. |
| `BTC-USD-dollar.csv`            | Barras alternativas de dólar generadas con la función `dollar_bars` para evitar el sesgo de calendario. |

Los experimentos que emplean barras alternativas calculan umbrales basados en la mediana del volumen en dólares para construir barras de tamaño homogéneo.

## Estructura del repositorio

```
btc_price_prediction/
├── data/                  # Datasets
├── images/                # Gráficos generados
├── afml/                  # Utilidades avanzadas (diferenciación fraccional)
├── data_utils/            # Utilidades de construcción de barras alternativas
├── robust/                # Métricas y validación robusta
├── regimes/               # Modelado de regímenes de mercado (KMeans/HMM)
├── config/                # Archivos de configuración y resultados JSON del pipeline
├── model.py               # Baseline: regresión lineal y Random Forest
├── compare_models.py      # Comparación de modelos clásicos (regresión y clasificación)
├── advanced_experiments.py    # Pruebas con indicadores técnicos y horizontes múltiples
├── enhanced_experiments.py    # Retorno logarítmico, purga temporal y métricas robustas
├── segmented_experiments.py   # Segmentación adaptativa por subseries
├── segment_agg_experiments.py # Segmentación con agregación de características
├── boosting_experiments.py    # Modelos XGBoost, LightGBM y CatBoost
├── rl_trading_drl.py          # Agente de refuerzo Q‑learning ligero con estado discreto
├── factory_pipeline.py        # Pipeline completo de la fábrica de estrategias cuantitativas【1794†source】
├── run_factory.py             # Lanzador del pipeline con configuración YAML
├── run_validation_regimes.py  # Validación CPCV y modelado de regímenes
└── ... (resultados *.txt y *.json)
```

### Módulos utilitarios

* **`afml/fdiff.py`** – funciones para diferenciación fraccional (*fractional differentiation*) según López de Prado【1794†source】.  Permite estacionar la serie sin perder memoria de largo plazo.
* **`data_utils/alt_bars.py`** – generación de barras de dólar, útiles para muestrear la serie según actividad monetaria y reducir sesgo temporal.
* **`robust/validation.py`** – implementación de la **Combinatorial Purged Cross‑Validation** (CPCV) con embargo temporal, siguiendo las directrices de validación robusta【1794†source】.
* **`robust/metrics.py`** – métricas de robustez como el ratio de Sharpe, deflated Sharpe ratio y probabilidad de *backtest overfitting* (PBO).
* **`regimes/model.py`** – modelo de regímenes basado en **KMeans** para identificar diferentes estados de mercado.  En ausencia de librerías de HMM, KMeans actúa como sustituto eficaz.

## Experimentos básicos

### 1. `model.py`: regresión lineal vs Random Forest

Implementa un pipeline sencillo para predecir el retorno diario utilizando 10 retardos de retornos.  Se calculan el error cuadrático medio (MSE) y la precisión direccional.  El modelo lineal logra MSE≈0,00097 y precisión ≈48,5 %, mientras que el Random Forest obtiene MSE≈0,00102 y precisión ≈45,9 %.  El script genera además el gráfico `images/cumulative_returns.png` que compara el retorno acumulado real y el predicho por el Random Forest.

### 2. `compare_models.py`: modelos clásicos de regresión y clasificación

Amplía los *features* con medias móviles y volatilidad.  Se evalúan modelos de regresión (Linear, RandomForest, ExtraTrees, GradientBoosting, SVR, KNN) y clasificación (LogisticRegression, RandomForestClassifier, SVC).  Ninguno supera claramente la barrera del 50 % de acierto, aunque la SVR se acerca a ~50,3 % y la LightGBM del script de boosting alcanza ≈53,7 %.

## Experimentos avanzados

### 3. `advanced_experiments.py`: indicadores técnicos y horizontes múltiples

Incluye RSI14, MACD, bandas de Bollinger y ATR14.  Evalúa horizontes de 1, 3, 5 y 10 días con un ExtraTreesRegressor y un esquema de validación temporal.  Reporta MSE, precisión, Net Points y Sharpe.  La precisión máxima observada es ≈53 % para 5 días, pero los **Net Points** predichos son negativos, indicando que las señales no generan ganancias netas.

### 4. `enhanced_experiments.py`: retorno logarítmico y purga temporal

Modifica la variable objetivo a retorno logarítmico y aplica validación cruzada con purga temporal (embargo).  Las métricas robustas (Net Points/ATR y Sharpe) muestran que la precisión ronda el 52 % y que los netos predichos siguen siendo negativos.  Útil para evaluar modelos con menor riesgo de sobreajuste.

### 5. `boosting_experiments.py`: XGBoost, LightGBM y CatBoost

Implementa estos potentes modelos de *gradient boosting* con los mismos *features* técnicos.  La **LightGBM** logra la mejor precisión (~53,7 % en 1 día) y el menor MSE, aunque la mejora frente a modelos de árboles tradicionales es limitada.

### 6. Segmentación adaptativa

Los últimos avances en **Temporal Fusion Transformers** proponen segmentar la serie en subseries dinámicas antes de predecir【699532840740462†L654-L713】.  Aquí se plasman dos enfoques:

1. **`segmented_experiments.py`**: Detecta incrementos del 3 %, 5 % o 10 % respecto al mínimo local, define segmentos alcistas/bajistas y entrena modelos `ExtraTreesRegressor` por categoría.  Con umbral 10 % y horizonte 10 días se logra precisión **≈61,7 %** y Sharpe ≈0,29【262867392780735†L1220-L1289】.
2. **`segment_agg_experiments.py`**: Segmenta la serie y resume cada subserie mediante estadísticas agregadas (retorno, volatilidad, promedios de indicadores, pendiente del log‑precio) para entrenar un `ExtraTreesClassifier`.  Se obtienen precisiones de 58–65 % según el umbral, pero los ratios de Sharpe son negativos, lo que sugiere señales direccionales sin rentabilidad.

## Reinforcement Learning

### 7. `rl_trading_drl.py`: agente Q‑learning discreto

Implementa un agente de Q‑learning basado en dos señales binarias: signo del retorno y cruce de medias móviles (MA3 vs MA7).  Explora 50 episodios con una política ε‑greedy y optimiza la Q‑tabla.  La política resultante (comprar tras días bajistas y vender tras días alcistas con MA3<MA7) genera un retorno acumulado ≈2,55 y un Sharpe anualizado ≈0,23, mostrando que incluso estrategias sencillas de RL pueden obtener rendimientos moderados.  El archivo de resultados `rl_results_drl.txt` contiene el resumen y `rl_equity_curve_drl.csv` la curva de equity.

> **Nota:** El antiguo script `rl_trading.py` ha sido eliminado por redundancia; se mantiene solo la versión mejorada.

## Pipeline avanzado: Fábrica de estrategias cuantitativas

### 8. `factory_pipeline.py` y `run_factory.py`

Estos módulos implementan un flujo completo de la *fábrica de estrategias* propuesto por López de Prado【1794†source】:

- **Generación de *features*** con nomenclatura unificada (`<símbolo>_<tf>_<feature>[_parámetros]`).  Incluye retornos, ATR, RSI, momentum, KAMA, StochRSI, volumen en dólares, *spread* y puede añadirse diferenciación fraccional y barras alternativas.
- **Etiquetado triple barrera** y **meta‑etiquetado** para filtrar señales.
- **Validación robusta** con CPCV y embargo temporal【1794†source】.
- **Detección de regímenes** mediante KMeans (o HMM si se dispone).
- **Backtest** con costes, slippage, volatility targeting y Kelly fraccional para dimensionar posiciones【1794†source】.
- **Métricas de robustez** como Sharpe, deflated Sharpe y PBO.

El script `run_factory.py` se ejecuta con una configuración YAML (por ejemplo `config/example.yaml`) y guarda los resultados en un JSON (`example.results.json`).  Un extracto de resultados con un conjunto de *features* básicos (`ret`, `atr_14`, `rsi_14`, `vol_usd`, `spread`) y horizonte 5 barras muestra: precisión ≈48,6 %, Sharpe ≈0,67, deflated Sharpe ≈0,61 y PBO ≈0,27.  Aunque el Sharpe es moderado, la precisión no supera el azar, reflejando la dificultad del problema y la necesidad de incorporar señales adicionales.

### 9. `run_validation_regimes.py`

Complementa el pipeline evaluando la aportación de modelos condicionados por regímenes de mercado.  Utiliza CPCV con 6 bloques y embargo de 5 observaciones, detecta 3 regímenes mediante KMeans y entrena tanto un modelo global como uno por régimen.  Los resultados (`config/validation_regimes.results.json`) indican que el modelo global supera ligeramente al modelo por regímenes en Sharpe y precisión, con PBO=0.0.

## Utilidades adicionales y configuraciones

Se incluyen distintos ficheros YAML en `config/` para experimentar con *features* y validaciones:

- `example.yaml` – configuración base para el pipeline.
- `example_ffd.yaml` – incluye la **diferenciación fraccional** como *feature* extra; sus resultados (`example_ffd.results.json`) muestran que la FFD no mejora el modelo con los parámetros elegidos.
- `example_alt.yaml` – emplea barras de dólar; los resultados (`example_alt.results.json`) muestran mayor volatilidad y PBO alto (0.8), sin mejora en precisión.
- `validation_regimes.results.json` – métricas comparativas del modelo global vs regímenes.

El repositorio también conserva ficheros `.txt` con resultados de cada experimento para referencia rápida (por ejemplo, `advanced_results.txt`, `boosting_results.txt`, `enhanced_results.txt`, `segmented_results.txt`, `segment_agg_results.txt`, `rl_results_drl.txt`).

## Archivos eliminados

Para evitar redundancias se han suprimido los siguientes ficheros:

- `rl_trading.py` y su salida `rl_results.txt`, reemplazados por `rl_trading_drl.py`.
- Todos los demás scripts se mantienen como referencia o comparativa; ninguno requiere dependencias externas no incluidas en este entorno.

## Cómo reproducir los experimentos

1. **Pruebas basales:**
   ```bash
   python model.py                     # regresión lineal y Random Forest
   python compare_models.py            # comparativa de modelos básicos
   ```

2. **Experimentos avanzados:**
   ```bash
   python advanced_experiments.py      # horizontes múltiples e indicadores técnicos
   python enhanced_experiments.py      # retorno logarítmico y purga temporal
   python boosting_experiments.py      # modelos de boosting
   python segmented_experiments.py     # segmentación adaptativa (umbral 3–10 %)
   python segment_agg_experiments.py   # segmentación con agregación de características
   ```

3. **Reinforcement Learning:**
   ```bash
   python rl_trading_drl.py            # agente Q‑learning sencillo
   ```

4. **Pipeline completo:**
   ```bash
   python run_factory.py --config config/example.yaml          # configuración básica
   python run_factory.py --config config/example_ffd.yaml      # con diferenciación fraccional
   python run_factory.py --config config/example_alt.yaml      # con barras de dólar
   ```

5. **Validación de regímenes:**
   ```bash
   python run_validation_regimes.py
   ```

> Asegúrate de ejecutar los comandos desde la raíz del repositorio (`btc_price_prediction`).  Todos los scripts guardan los resultados en ficheros `.txt` o `.json` en el mismo directorio para facilitar su inspección.

## Conclusiones y próximos pasos

El conjunto de pruebas aquí recopilado evidencia que, con datos diarios de BTC‑USD y *features* técnicos básicos, **ningún modelo supera consistentemente el 55 % de aciertos direccionales**.  Las técnicas de segmentación y boosting logran mejoras puntuales (hasta ~62 % de aciertos y Sharpe ~0,29) pero a menudo no se traducen en beneficios netos sostenibles.  El pipeline avanzado, aunque robusto, muestra que la estrategia base necesita señales adicionales para superar la pura aleatoriedad.  La incorporación de **indicadores on‑chain**, **volumen en barras alternativas**, **análisis de sentimiento** y la experimentación con modelos de atención (Transformers) o reinforcement learning profundo son líneas naturales de investigación futura【699532840740462†L786-L894】.  Asimismo, las utilidades de diferenciación fraccional y barras de dólar ofrecen nuevas vías para mejorar la estacionariedad y el muestreo de la serie sin modificar la estructura general del pipeline.
