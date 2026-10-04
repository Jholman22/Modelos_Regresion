# 🏠 Comparación de Modelos de Regresión — Precio de Viviendas (Ames Housing)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jholman22/Parcial1_TAM/blob/main/TAM_PARCIAL_PUNTO_2_%26_3_JD.ipynb)

Proyecto del **Parcial 1 de TAM (puntos 2 y 3)**. Se predice el precio de venta de viviendas (`SalePrice`) con el conjunto de datos **Ames Housing**, se comparan **9 modelos de regresión** con distintas estrategias de optimización de hiperparámetros y se presentan los resultados en un **dashboard interactivo con Streamlit**.

---

## 📌 Descripción

1. **Preprocesamiento** con un transformador propio (`mypre_ames`) que imputa valores faltantes y codifica variables categóricas.
2. **Análisis exploratorio:** matriz de correlación, boxplots y *scatter matrix* de las variables más relacionadas con el precio.
3. **Entrenamiento y optimización** de 9 regresores usando **Optuna** (búsqueda aleatoria, *grid search* y optimización bayesiana TPE).
4. **Evaluación** con MAE, MSE, R² y MAPE.
5. **Dashboard** en Streamlit para comparar modelos.

## 📂 Datos

- **Dataset:** Ames Housing (descarga desde Google Drive dentro del notebook)
- **Variable objetivo:** `SalePrice`
- **Partición:** 70 % entrenamiento / 30 % prueba
- **Columnas eliminadas** por exceso de valores faltantes o poca utilidad: `Order`, `PID`, `Alley`, `Pool QC`, `Fence`, `Misc Feature`, `Misc Val`, `Fireplace Qu`, `Garage Yr Blt`, `3Ssn Porch`, `Screen Porch`, `Pool Area`, `Mo Sold`

## ⚙️ Preprocesamiento

El transformador `mypre_ames` (compatible con `Pipeline` de scikit-learn) hace lo siguiente:

| Tipo de variable | Imputación | Codificación |
|---|---|---|
| Numéricas | Mediana | — |
| Categóricas (38 columnas) | Moda | `OrdinalEncoder` con categorías ordenadas por frecuencia; categorías desconocidas → `-1` |

Después se aplica un escalador (`RobustScaler` en casi todos los modelos, `StandardScaler` en SGD).

## 🤖 Modelos y optimización

| Modelo | Hiperparámetros ajustados | Estrategias |
|---|---|---|
| Regresión lineal | — (línea base) | Validación cruzada 5-fold |
| Lasso | `alpha` | Random → Grid → Bayesiana |
| ElasticNet | `alpha`, `l1_ratio` | Random → Grid → Bayesiana |
| Kernel Ridge (RBF) | `alpha`, `gamma` | Random → Grid → Bayesiana |
| SGDRegressor | `penalty` y otros | Random → Grid → Bayesiana |
| Bayesian Ridge | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` | Random → Grid → Bayesiana |
| Gaussian Process Regressor | `sigma_0` (kernel lineal) | Random → Grid → Bayesiana |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_*`, `max_features` | Random → Grid → Bayesiana |
| SVR | `kernel`, `C`, `epsilon` | Random → Grid → Bayesiana |

Para cada modelo, la búsqueda aleatoria estima la importancia de cada hiperparámetro, y las búsquedas siguientes se concentran en una zona más estrecha alrededor del mejor valor. La métrica de optimización es el **MAE con validación cruzada de 5 particiones**.

## 📊 Resultados (conjunto de prueba)

| Modelo | MAE | R² | MAPE |
|---|---|---|---|
| **Random Forest** | **16,277** | **0.90** | **9.59 %** |
| Kernel Ridge | 19,673 | 0.863 | 11.46 % |
| SVR | 19,909 | 0.81 | 10.87 % |
| ElasticNet | 20,789 | 0.813 | 11.75 % |
| Bayesian Ridge | 20,945 | 0.81 | 11.94 % |
| Lasso | 21,010 | 0.812 | 12.03 % |
| SGDRegressor | 21,021 | 0.81 | 12.03 % |
| Gaussian Process | 21,021 | 0.81 | 12.04 % |
| Regresión lineal | 21,021 | 0.812 | 12.04 % |

**Conclusiones principales**

- **Random Forest** es el mejor modelo, con un error promedio de unos 16,000 dólares y R² ≈ 0.90. Es un resultado sólido, aunque se observa cierto sobreajuste (R² de 0.98 en entrenamiento frente a 0.90 en prueba).
- **Kernel Ridge** es el mejor de los modelos regularizados y supera a todos los lineales.
- Los **modelos lineales regularizados** (Lasso, ElasticNet, Bayesian Ridge, SGD) rinden casi igual que la regresión lineal simple: la regularización aporta poco en este problema.
- Afinar los hiperparámetros produjo mejoras pequeñas; la elección de la **familia de modelo** pesó mucho más.

## 📈 Dashboard (Streamlit)

La aplicación `app_streamlit.py` tiene tres pestañas:

- **Comparación:** gráfico de barras por métrica y destacado automático de los 3 mejores modelos (combinando R² alto y MAPE bajo), con detalle por modelo.
- **Mapa de correlación:** matriz de correlación de las variables preprocesadas.
- **Dataset:** tabla del conjunto de datos preprocesado.

Se lanza desde Colab con `streamlit` y un túnel `ngrok`. Para ejecutarlo en local:

```bash
pip install streamlit pandas plotly pillow
streamlit run app_streamlit.py
```

Archivos necesarios: `metricas.csv`, `Dataset.csv` y `logoun.png`.

## 🚀 Cómo ejecutarlo

1. Abre el notebook en Colab con el botón de arriba.
2. Ejecuta las celdas en orden. Se descarga el dataset, se entrenan los modelos y se generan `metricas.csv` y `Dataset.csv`.
3. Para el dashboard, ejecuta las celdas finales (o usa la instrucción local anterior).

```bash
pip install pandas numpy seaborn matplotlib scikit-learn optuna plotly streamlit
```

## ⚠️ Limitaciones y mejoras posibles

- **Partición sin semilla:** `train_test_split` no fija `random_state`, por lo que los resultados cambian en cada ejecución. Conviene fijarlo para reproducibilidad.
- **Ajuste en subconjunto:** Random Forest y Gaussian Process se optimizan con solo 300 muestras, lo que puede dar hiperparámetros poco óptimos.
- **Sobreajuste del Random Forest:** la diferencia entre entrenamiento y prueba es grande; conviene limitar la profundidad o aumentar `min_samples_leaf`.
- **Codificación ordinal por frecuencia:** impone un orden artificial a categorías sin jerarquía (por ejemplo `Neighborhood`). Probar *One-Hot* o *Target Encoding* podría ayudar a los modelos lineales. Además, el orden de categorías se calcula con todo el dataset antes de dividir.
- **Dashboard:** el R² se muestra con signo `%` aunque es una razón entre 0 y 1.
- **Modelos más potentes:** no se probaron *gradient boosting* (LightGBM, XGBoost), que suelen ganar en este tipo de datos.
- **Objetivo en escala logarítmica:** transformar `SalePrice` con `log` suele mejorar el ajuste de los modelos lineales.

## 🛠️ Tecnologías

Python · pandas · NumPy · scikit-learn · Optuna · Matplotlib · Seaborn · Plotly · Streamlit · ngrok · Google Colab

## 👤 Autor

**Jholman22** — [GitHub](https://github.com/Jholman22)
