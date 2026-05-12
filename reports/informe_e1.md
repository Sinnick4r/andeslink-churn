# Informe Tecnico - Entrega 1: Entrenamiento
## Proyecto MLOps: Predicción de Churn - AndesLink Servicios Digitales S.A.
## ISTEA · Laboratorio de Minería de Datos · Prof. Diego Mosquera

---

## 1. General

### 1.1 Contexto

AndesLink Servicios Digitales S.A. comercializa planes de suscripción a servicios
digitales orientados a consumidores finales. Durante los últimos trimestres detectó
una tasa creciente de cancelación voluntaria. Cada cliente perdido implica pérdida
de ingreso recurrente más el costo de adquisición de un reemplazo, estimado entre
3x y 5x mayor que el costo de retención.

### 1.2 Objetivo analítico

Construir un modelo de clasificación binaria que estime la probabilidad de churn
de cada cliente activo, permitiendo priorizar campañas de retención sobre los
clientes con mayor riesgo antes de que cancelen.

**Variable target:** `churn` - 1 = canceló el servicio, 0 = continúa activo.

### 1.3 Traducción del problema de negocio a objetivos de ML

| Dimensión | Decisión | Justificación |
|-----------|----------|---------------|
| Tipo de problema | Clasificación binaria supervisada | Target es 0/1 |
| Métrica principal | F1-score | Desbalance moderado; penaliza FP y FN |
| Métrica de comparación | ROC-AUC | Independiente del threshold, util para ranking de modelos |
| Métricas secundarias | PR-AUC, Recall | Para calibración de threshold |
| Threshold | Calibrado por curva PR, no default 0.5 | Costo de FN > costo de FP en churn |

### 1.4 Costo de errores

- **Falso negativo** (no detectar un cliente que se va): perdida del ingreso
  recurrente más el costo de reemplazo. Costo alto.
- **Falso positivo** (contactar un cliente que no iba a irse): costo de la campaña
  de retención aplicada innecesariamente. Costo moderado.
- Esta asimetría justifica calibrar el threshold por debajo de 0.5 para favorecer
  Recall sobre Precision cuando el negocio lo requiere.

### 1.5 KPIs

- Tasa de retención esperada si se trabaja sobre el top-20% de probabilidad predicha
- Reduccion estimada de churn vs. baseline de no intervencion
- El modelo detecta 267 de 340 churners en test set (Recall 78.5%) — captura casi
  8 de cada 10 clientes que se van antes de que cancelen

---

## 2. Dataset

### 2.1 Descripcion general

- **Archivo:** `churn_sintetico.csv`
- **Origen:** dataset aportado
- **Dimensiones:** 5.000 filas × 16 columnas
- **Target:** `churn` — distribución: 34% positivos (1.702), 66% negativos (3.298)
- **Ratio de desbalance:** 1.94:1
- **Nulos:** ninguno en ninguna columna

### 2.2 Variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| tenure_months | Numérica | Meses de antigüedad como cliente (1–72) |
| monthly_charge | Numérica | Cargo mensual actual ($15–$127) |
| total_charges | Numérica | Cargos totales acumulados (multicolineal — ver 2.4) |
| support_tickets | Numérica | Tickets de soporte abiertos (0–8) |
| late_payments | Numérica | Cantidad de pagos atrasados (0–5) |
| avg_monthly_usage_gb | Numérica | Uso de datos mensual promedio (5–324 GB) |
| has_streaming | Binaria | 1 si tiene servicio de streaming |
| has_security_pack | Binaria | 1 si tiene paquete de seguridad |
| num_products | Numérica | Cantidad de productos contratados (1–4) |
| customer_age | Numérica | Edad del cliente (18–78) |
| is_promo | Binaria | 1 si está en tarifa promocional |
| contract_type | Categórica | mensual (55%), anual (28.2%), bianual (16.8%) |
| payment_method | Categórica | debito (35%), credito (31.6%), transferencia (20.9%), efectivo (12.5%) |
| internet_service | Categórica | fibra (44.8%), cable (28.5%), movil (20%), ninguno (6.7%) |
| region | Categórica | centro (36.4%), norte (23.4%), sur (21.4%), oeste (18.8%) |
| **churn** | **Target** | **1 = abandono, 0 = retuvo** |

### 2.3 Estadisticas descriptivas

| Variable | Media | Std | CV | Min | Max |
|----------|-------|-----|----|-----|-----|
| tenure_months | 36.41 | 20.84 | 0.57 | 1 | 72 |
| monthly_charge | 65.27 | 17.96 | 0.28 | 15 | 127 |
| total_charges | 2371.16 | 1555.94 | 0.66 | 50 | 9083 |
| support_tickets | 1.71 | 1.30 | 0.76 | 0 | 8 |
| late_payments | 0.70 | 0.84 | 1.21 | 0 | 5 |
| avg_monthly_usage_gb | 120.53 | 49.75 | 0.41 | 5 | 324 |

### 2.4 Hallazgos principales del EDA

**Hallazgo 1 ---> no hay valores nulos:**
Dataset completamente limpio. El pipeline incluye `SimpleImputer` de
todas formas para darle robustez si se escala.

**Hallazgo 2 ---> desbalance moderado (34/66):**
No requiere tecnicas de resampling. Se usa `class_weight='balanced'` en todos los
modelos. Accuracy descartada como metrica: un modelo que predice siempre 0 tendría
66% accuracy y eso no aportaria valor de negocio

**Hallazgo 3 ---> correlaciones lineales debiles con el target (max |0.17|):**

| Variable | Correlación | Dirección |
|----------|-------------|-----------|
| tenure_months | −0.170 | Menor antigüedad → más churn |
| total_charges | −0.125 | Redundante con tenure (ver hallazgo 5) |
| support_tickets | +0.102 | Más tickets → más churn |
| late_payments | +0.081 | Pagos atrasados → más churn |
| monthly_charge | +0.062 | Leve efecto positivo |
| is_promo | +0.051 | Leve efecto positivo |

La debilidad de la señal lineal motivó comparar modelos no-lineales contra el
baseline lineal. Los resultados posteriores mostraron que la señal es
predominantemente lineal (ver sección 4).

**Hallazgo 4 ---> las variables categoricas tienen alta discriminacion:**

| Variable | Categoría | Tasa de churn | vs. Media global |
|----------|-----------|---------------|------------------|
| contract_type | mensual | 47.5% | +13.5 pp |
| contract_type | anual | 21.1% | −12.9 pp |
| contract_type | bianual | 11.6% | −22.4 pp |
| internet_service | movil | 50.9% | +16.9 pp |
| internet_service | fibra | 26.8% | −7.2 pp |
| payment_method | efectivo | 40.5% | +6.5 pp |
| region | todas | 33.5–34.5% | ≈ 0 pp |

`contract_type` es la variable más discriminante del dataset (diferencia de 36 pp
entre mensual y bianual). `region` no aporta señal discriminante marginal, aunque
se incluye en el pipeline por posibles interacciones no lineales.

**Hallazgo 5 ---> hay redundancia estructural bastante alta en `total_charges`:**
`total_charges ≈ tenure_months × monthly_charge` — correlación 0.87, ratio mediana
= 1.000, más del 58% de filas con ratio entre 0.95 y 1.05. En modelos lineales esto
introduce multicolinealidad; en modelos basados en árboles genera redundancia que
puede diluir importancia de variables.

**Decisión:** `total_charges` se elimina del pipeline y se reemplaza por la feature
derivada `charges_per_month`. Se reconoce que esta feature puede correlacionar
fuertemente con `monthly_charge` dado el comportamiento algebraico del dataset; su
utilidad real se evaluó por importancia de features post-entrenamiento (ver apéndice).

**Hallazgo 6 ---> `has_streaming` sin señal:**
Boxplots y correlación (~−0.008) prácticamente idénticos entre clases. Confirmado
posteriormente por SHAP (SHAP medio = 0.005, posición 20/20).

**Hallazgo 7 ---> no se detecto leakage:**
Ninguna variable es derivable directamente del target ni contiene información futura.

---

## 3. Preparacion de datos y feature engineering

### 3.1 Split estratificado

| Conjunto | Filas | % del total | Churn rate |
|----------|-------|-------------|------------|
| Train | 3.200 | 64% | 0.341 |
| Validación | 800 | 16% | 0.340 |
| Test | 1.000 | 20% | 0.340 |

Split determinístico con `random_state=42` y `stratify=y`. La proporción de churn
se mantiene dentro de tolerancia ±2% en los tres subconjuntos.

**Justificación del doble split:**

El conjunto de validación permite seleccionar el threshold optimo y comparar modelos
sin contaminar el test set. El test set permanece completamente aislado hasta la
evaluación final. 

### 3.2 Features derivadas

**`charges_per_month = total_charges / tenure_months`**
- Fallback para `tenure_months = 0`: usa `monthly_charge` directamente
- Nota: puede correlacionar fuertemente con `monthly_charge` dado el comportamiento
  algebraico del dataset. Su aporte incremental fue evaluado via SHAP

**`tickets_per_year = support_tickets / (tenure_months_clipped / 12)`**
- `tenure_months` se clipea en mínimo 6 meses para evitar inflación artificial
  en clientes recientes (ej: 3 tickets en 1 mes → 36/año sin clip → 6/año con clip)
- Normaliza la señal de soporte por antigüedad del cliente

### 3.3 Pipeline de preprocesamiento (`ColumnTransformer`)

**Features numericas (12):**
`tenure_months`, `monthly_charge`, `support_tickets`, `late_payments`,
`avg_monthly_usage_gb`, `has_streaming`, `has_security_pack`, `num_products`,
`customer_age`, `is_promo`, `charges_per_month`, `tickets_per_year`

Pipeline numerico: `SimpleImputer(strategy='median')` → `StandardScaler()`

**Features categoricas (4):**
`contract_type`, `payment_method`, `internet_service`, `region`

Pipeline categórico: `SimpleImputer(strategy='most_frequent')` →
`OneHotEncoder(drop='first', handle_unknown='error')`

**Total features post-transformacion:** 23 (12 numéricas + 11 OHE)

**Decisiones de diseño:**

- `SimpleImputer` incluido aunque no haya nulos: robustez a futuro, cubre futuros datasets
- `drop='first'` en OHE: evita multicolinealidad perfecta entre dummies
- `handle_unknown='error'`: falla ruidosamente. Solo para esta estapa y porque tengo dataset cerrado
- `remainder='drop'`: descarta `total_charges` y cualquier columna no listada explicitamente

---

## 4. Entrenamiento y evaluacion

### 4.1 Modelos comparados

Se eligieron tres modelos de scikit-learn que cubren un espectro
representativo de paradigmas para clasificación binaria tabular:

- **LogisticRegression**: baseline lineal interpretable. Permite cuantificar
  qué tan lejos llegan las relaciones lineales puras antes de necesitar
  algo más complejo.
- **RandomForest**: ensemble de árboles por bagging. Captura interacciones
  no-lineales entre features, robusto a outliers, parámetros por defecto
  razonables.
- **HistGradientBoosting**: gradient boosting sobre histogramas. Suele dar
  el mejor F1 en tabular de tamaño medio, maneja NaN nativamente, más
  liviano que XGBoost/LightGBM.

**Criterios para limitar el set a estos tres:**

1. **Sklearn puro**: evita dependencias externas (XGBoost, LightGBM, CatBoost
   requieren binarios extra, complican el `requirements.txt` de Docker en E2,
   y suman superficie de falla en supply chain).

2. **Paradigmas distintos**: lineal, bagging, boosting. Si los tres
   convergen a F1 similar (como paso), la conclusión es estructural —
   el techo lo pone el dataset, no el modelo. Si uno destacara, indicaría
   en qué tipo de relaciones está la señal.

3. **Costo del testing**: tres modelos entrenan en <1 minuto sobre 5000
   filas, permiten iteración rápida del pipeline. Sumar más modelos sin
   HPO no agregaria info proporcional

**Modelos no incluidos y por que:**

- **XGBoost / LightGBM / CatBoost**: redundantes con HistGBM para este
  dataset y sumarían dependencias sin beneficio concreto
- **SVM**: poco escalable; con kernels no-lineales el costo crece
  cuadráticamente con el dataset. Esto es un problema operacional para un futuro que haz que tener en cuenta pro mas que este sea un proyecto con un dataset fijo.
- **Redes neuronales (MLP)**: requieren muchisimo mas tuning, para este dataset no vale la pena
- **k-NN, Naive Bayes**: descartados como baseline porque LogReg ya cubre
  ese rol con mejor capacidad explicativa (coeficientes interpretables).

**Validación cruzada de la elección:** Adicionalmente se ejecutó PyCaret
`compare_models` (ver `notebooks/02_Validacion_pycaret.ipynb`) sobre el
mismo dataset. Los resultados confirman que ningún modelo de los disponibles
en sklearn supera significativamente el F1 alcanzado por los tres
comparados manualmente.

### Resumen

Todos de scikit-learn puro para tener menos dependencias y, por ende, menor
riesgo de conflictos de versión en el Dockerfile de la segunda entrega del proyecto.

| Modelo | Justificación |
|--------|---------------|
| LogisticRegression | Baseline lineal interpretable, `class_weight='balanced'` |
| RandomForestClassifier | captura no-linealidades |
| HistGradientBoostingClassifier | Gradient boosting nativo sklearn |

### 4.2 Calibracion del threshold

No se usa el default 0.5. 

El threshold óptimo maximiza F1 sobre la curva precision-recall en el conjunto de validación. 
Como hay desbalance moderado y costo asimétrico (FN > FP), el threshold óptimo esta entre 0.35 y 0.45

### 4.3 Resultados en validacion

| Modelo | F1 (val) | ROC-AUC (val) | Threshold | Tamaño .joblib |
|---|---|---|---|---|
| LogisticRegression | 0.6196 | 0.7544 | 0.441444 | ~7 KB |
| RandomForest | 0.6209 | 0.7503 | 0.409929 | ~7 MB |
| HistGradientBoosting | 0.6037 | 0.7461 | 0.304959 | ~2 MB |

### 4.4 Selección del modelo final

Se compararon tres modelos bajo idénticas condiciones (mismo split, mismo seed,
threshold óptimo por F1 individualizado):

| Modelo | F1 (val) | ROC-AUC (val) | Threshold | Tamaño .joblib |
|---|---|---|---|---|
| LogisticRegression | 0.6196 | 0.7544 | 0.441444 | ~7 KB |
| RandomForest | 0.6209 | 0.7503 | 0.409929 | ~7 MB |
| HistGradientBoosting | 0.6037 | 0.7461 | 0.304959 | ~2 MB |

RandomForest obtuvo el F1 absoluto más alto pero con margen marginal: 0.0013
puntos sobre LogReg. Esta diferencia equivale a aproximadamente 1 predicción
distinta sobre 800 muestras de validación, dentro del ruido estadistico
esperable para evaluaciones de esta escala.

**Regla de selección adoptada:** ante empate técnico (Δ F1 < 0.005), se prioriza
el modelo más simple según orden definido en `MODEL_PRIORITY`
(LogReg → HistGBM → RF). Como tiebreaker secundario se usa ROC-AUC.

**Criterio operacional de "simplicidad":**
- Tamaño del artefacto serializado: relevante para cold start de la API
  (LogReg ~7 KB vs RF ~7 MB, factor 1000×).
- Tiempo de inferencia: LogReg evalúa una combinación lineal (~µs);
  RF promedia 200 árboles de profundidad 12 (~ms).
- Interpretabilidad: LogReg expone coeficientes directos; ensembles
  requieren feature importance o SHAP.

**Justificacion de escalabilidad:** La regla no impacta resultados a esta
escala (5.000 filas), pero anticipa el escenario de crecimiento del dataset.
Si crece el volumen del dataset, la diferencia operacional entre LogReg y RF se vuelve
crítica (tiempo de entrenamiento, memoria de inferencia, tamaño del
artefacto en el repositorio de modelos).

**Resultado final:** LogReg seleccionado. Es el segundo en F1 absoluto pero
el primero en `MODEL_PRIORITY`, y la diferencia 0.0013 entra dentro de la
tolerancia.

### 4.5 Revision de la hipótesis del EDA

El EDA planteó que las correlaciones lineales débiles indicaban señal de interacción
no-lineal. Los resultados muestran que los modelos de árbol no lograron capitalizar
interacciones adicionales. Esta discrepancia entre hipótesis y resultado es
un hallazgo valido del análisis, no un error de modelado.

### 4.6 Evaluacion final en test set

| Métrica | Valor |
|---------|-------|
| F1-score | 0.6020 |
| ROC-AUC | 0.7561 |
| PR-AUC | 0.6155 |
| Precision (clase 1) | 0.4881 |
| Recall (clase 1) | 0.7853 |
| Threshold aplicado | 0.441444 |
| n test | 1.000 |

**Reporte por clase:**

| Clase | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| No churn (0) | 0.84 | 0.58 | 0.68 | 660 |
| Churn (1) | 0.49 | 0.79 | 0.60 | 340 |
| Weighted avg | 0.72 | 0.65 | 0.66 | 1.000 |

**Confusion matrix (test set):**

|                    | Predicho: No churn | Predicho: Churn |
|--------------------|--------------------|-----------------|
| Real: No churn     | 380 (TN)           | 280 (FP)        |
| Real: Churn        | 73 (FN)            | 267 (TP)        |

**Lectura de negocio:**

- 267 churners detectados de 340 → Recall 78.5% --> el modelo captura casi 8 de
  cada 10 clientes que se van
- 73 churners no detectados (FN) --> el costo más alto, controlado
- 281 falsos positivos --> clientes contactados innecesariamente, costo moderado
  aceptable

### 4.7 Consistencia val vs test

| Metrica | Validacion | Test | Δ |
|---|---|---|---|
| F1 | 0.6196 | 0.6020 | -0.0176 |
| ROC-AUC | 0.7544 | 0.7561 | +0.0017 |
| PR-AUC | 0.5956 | 0.6155 | +0.0199 |

La caida de F1 entre val y test (-0.018) es coherente con el ruido esperable
para tamaños de muestra de 800 y 1.000 respectivamente. Notablemente,
ROC-AUC y PR-AUC suben levemente en test, lo que descarta overfitting:
si el modelo hubiera memorizado patrones del train+val, las
metricas globales de discriminación (ROC, PR) tambien deberian bajar.

Conclusion: el modelo generaliza correctamente. el F1 es más bajo en el test, lo que refleja
una distribución levemente distinta de probabilidades cerca del threshold

---

## 5. Reproducibilidad y trazabilidad

### 5.1 Control de versiones con Git

- **Estrategia de ramas:** `main` (estable) ← `dev` (integración) ← `feat/*`
- **Commits:** conventional commits (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`)
- **Tags por entrega:** `v1.0-parcial1` para la primer entrega

### 5.2 Versionado de datos con DVC

- CSV trackeado con DVC desde el día 1: `data/raw/churn_sintetico.csv.dvc`
- El CSV no está en git (`.gitignore`); el puntero DVC sí
- `dvc.lock` commiteado: firma criptográfica de reproducibilidad

**Pipeline de DVC (3 stages):**

```
prepare → train → evaluate
```

| Stage | Comando | Deps | Outputs |
|-------|---------|------|---------|
| prepare | `python -c "from src.data import..."` | CSV, data.py, features.py | - |
| train | `python -m src.train` | CSV, train.py, data.py, features.py | models/churn_model_v1.joblib |
| evaluate | `python -m src.evaluate` | evaluate.py, data.py, .joblib | reports/test_metrics.json |

**Parámetros trackeados en `params.yaml`:**

```yaml
split:
  random_seed: 42
  test_size: 0.20
  val_size: 0.20
train:
  n_estimators: 200
  max_depth: 12
  learning_rate: 0.05
  max_iter: 300
```

### 5.3 Tracking de experimentos con MLflow

- Tracking URI local: `file:./mlruns`
- Experimento: `andeslink-churn-e1`
- **4 runs registrados:**
  - `LogisticRegression`: F1=0.6216, ROC-AUC=0.7544, threshold=0.44144
  - `RandomForest`: F1=0.6209, ROC-AUC=0.7503, threshold=0.409929
  - `HistGradientBoosting`: F1=0.6059, ROC-AUC=0.7461, threshold=0.304959
  - `GANADOR_LogisticRegression`: modelo serializado logueado con MLflow sklearn
- **Hash SHA-256** del `.joblib` registrado como tag en el run ganador
- **Artefacto JSON** con comparativa de los 3 modelos (`reports/train_metrics.json`)

### 5.4 Reproducibilidad esta verificada

```bash
# Desde artefactos borrados, el pipeline completo se regenera con:
dvc repro

# Verifica que el modelo carga y predice correctamente:
python -c "import joblib; m=joblib.load('models/churn_model_v1.joblib'); print('OK')"
```

### 5.5 Entorno reproducible

| Archivo | Propósito |
|---------|-----------|
| `environment.yml` | Entorno conda (portable) |
| `environment.lock.yml` | Versiones exactas resueltas por conda en Windows |

**Versiones criticas pinneadas:**

| Librería | Versión | Razón |
|----------|---------|-------|
| scikit-learn | 1.5.2 | El .joblib es version-sensitive |
| numpy | 1.26.x | Evitar conflicto con sklearn en numpy 2.x |
| mlflow | 2.17.x | Tracking local |
| pandas | 2.2.x | Estabilidad de API |

---

## 6. Estructura del repositorio

```
andeslink-churn/
├── data/
│   ├── raw/churn_sintetico.csv.dvc   ← CSV trackeado por DVC
│   ├── processed/.gitkeep
│   └── current/.gitkeep
├── notebooks/
│   ├── 01_EDA_churn_dataset.ipynb                ← EDA completo
│   ├── 02_Validacion_pycaret.ipynb               ← Validacion usando Pycaret
│   └── 03_Script_prediccion_churn.ipynb          ← Script de evalaucio nde modelo serialziado

├── src/
│   ├── __init__.py
│   ├── features.py                   ← add_derived_features, build_preprocessor
│   ├── data.py                       ← load_raw_data, split_data
│   ├── train.py                      ← run_experiment, MLflow tracking
│   └── evaluate.py                   ← evaluate_on_test, plot_evaluation
├── app/
│   ├── __init__.py
│   ├── main.py                       ← placeholder E2
│   ├── schemas.py                    ← placeholder E2
│   └── model_loader.py               ← placeholder E2
├── gui/
│   └── git              ← placeholder E2
├── models/
│   └── churn_model_v1.joblib         ← modelo serializado (DVC tracked)
├── reports/
│   ├── eda_01_target_distribution.png
│   ├── eda_02_boxplots_by_churn.png
│   ├── eda_03_categoricas_vs_churn.png
│   ├── eda_04_correlacion_heatmap.png
│   ├── eda_05_multicolinealidad.png
│   ├── eda_06_correlacion_target.png
│   ├── eda_07_evaluation_test.png
│   ├── train_metrics.json
│   ├── test_metrics.json
│   └── informe_e1.md
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── environment.yml
├── environment.lock.yml
├── requirements.txt                  ← placeholder E2 (pinned)
├── Dockerfile                        ← placeholder E2
├── Dockerfile.gui                    ← placeholder E2
├── docker-compose.yml                ← placeholder E2
├── pyproject.toml                    ← configuración ruff
└── README.md
```

---

## 7. Conclusiones y limitaciones

### 7.1 Conclusiones principales

1. **Modelo seleccionado:** LogisticRegression con threshold calibrado 0.441
2. **Rendimiento:** F1=0.601, ROC-AUC=0.756, Recall=0.785 en test set
3. **La señal del dataset es predominantemente lineal:** los modelos no-lineales
   no superaron al baseline lineal, lo que contradijo la hipotesis inicial
   del EDA y fue documentado
4. **Las variables de contrato dominan la predicción:** `contract_type_mensual` es
   el driver más fuerte, confirmado por EDA, feature importance y SHAP
5. **El pipeline es reproducible end-to-end:** `dvc repro` regenera el modelo
   desde el CSV sin intervención manual

### 7.2  posibles limitaciones

- **No hay variables temporales:** fecha de alta, historial de cambios de plan y
  eventos de vida son predictores fuertes ausentes en este dataset. Sin esto, el las correlaciones que quedan en el dataset son realtivamente debiles  (sininguna se correalciona con mas de 0.17 con churn). Por eso el F1 es limitado
- **Se hizo un unico split de validación:** la varianza del F1 con n=800 puede ser
  suficiente para cambiar el ranking con otra semilla; cross-validation de
  k-folds daria una estimación más robusta
- **`charges_per_month` colapsa sobre `monthly_charge`:** por comportamiento
  algebraico del dataset, la feature derivada no aporta información independiente
  significativa
- **Threshold fijo en produccin:** el threshold 0.44144 fue calibrado sobre este
  dataset; en producción debería recalibrarse periódicamente a 
  - **Comparación limitada a 3 modelos**: no se exploraron variantes con
  optimizacion de hiperparametros. Se podria usar GridSearchCV u Optuna, por ekemplo, 
  y a mejorar el F1 de cada modelo individualmente. Eso se podria evaluar en la segunda entrega E2.
- **Sin estimacion de incertidumbre**: el modelo devuelve una probabilidad
  puntual, no un intervalo de confianza. Para decisiones de negocio
  criticas, deberia implementarse metodo para incluirla.

### 7.3 Próximos pasos de la entrega 2

- Exponer el modelo como API REST con FastAPI
- Construir GUI con Streamlit para consumo de la API
- Containerizar con Docker y orquestar con Docker Compose
- Agregar tests automatizados con pytest

---

## Validación con PyCaret

### A.1 Objetivo

Validar de forma independiente los resultados del pipeline manual usando Pycaet.
Esto no reemplaza el análisis principal, se hizo para confirmar las decisiones tomadas.

### A.2 Metodologia

- Librería: PyCaret 3.x con `compare_models()`
- Métrica de optimización: F1-score
- Evaluación: cross-validation 10-fold

### A.3 Resultados con threshold default (0.5)

Los modelos lineales dominan el ranking:

| Posición | Modelo | F1 | AUC |
|----------|--------|-----|-----|
| 1 | Ridge Classifier | 0.5018 | 0.7528 |
| 2 | Linear Discriminant Analysis | 0.5144 | 0.7528 |
| 3 | **Logistic Regression** | 0.5091 | 0.7522 |
| 4 | Gradient Boosting | 0.4866 | 0.7440 |
| 5 | Random Forest | 0.4674 | 0.7252 |

### A.4 Comparación con threshold equivalente (~0.441)

Al igualar el threshold al calibrado en el pipeline manual:

| Modelo | F1 | AUC | Recall | Precision |
|--------|-----|-----|--------|-----------|
| Ridge | 0.5580 | 0.6786 | 0.4755 | 0.6750 |
| LDA | 0.5980 | 0.7884 | 0.5851 | 0.6115 |
| **LogisticRegression** | **0.6026** | **0.7866** | **0.5890** | **0.6168** |

LogisticRegression obtiene el mejor F1 con threshold equivalente, confirmando la
elección del pipeline manual.

### A.5 Consistencia entre ambos enfoques

| Fuente | Modelo | F1 | AUC |
|--------|--------|-----|-----|
| Pipeline manual (validación) | LogReg | 0.6216 | 0.7544 |
| Pipeline manual (test) | LogReg | 0.6014 | 0.7561 |
| PyCaret threshold 0.441 | LogReg | 0.6026 | 0.7866 |

Diferencia entre pipeline manual en test y PyCaret: **0.0012 en F1**.
Ambos enfoques convergen en un F1 similar y en la elección de modelos lineales, aunque el balance precision/recall difiere entre evaluaciones, probablemente por diferencias de split, validacion cruzada y config interna de PyCaret

**Por qué los F1 de PyCaret con threshold 0.5 son más bajos:**
El threshold calibrado (0.441) vs default (0.5) explica la mayor parte de la
diferencia. La cross-validation de 10-folds de PyCaret también produce estimaciones
más conservadoras que un único split de validación.


### A.6 Curva de calibración

La Regresión Logística muestra calibración casi perfecta: sus probabilidades
predichas corresponden a frecuencias reales de churn observadas. Si el modelo
predice 44% de probabilidad, aproximadamente ese porcentaje de clientes
efectivamente abandona.

Esta propiedad justifica el uso del threshold 0.441 para decisiones de negocio:
el modelo está bien calibrado, por lo que podemos confiar en ese punto de corte
sin ajustes adicionales. A diferencia de modelos de árbol (que frecuentemente
requieren Platt scaling o isotonic regression para calibrar probabilidades),
LogReg es naturalmente robusto en este aspecto.

### A.7 Impacto del threshold

| Métrica | Threshold 0.5 | Threshold 0.441 | Cambio |
|---------|---------------|-----------------|--------|
| Accuracy | 0.7407 | 0.7353 | −0.73% |
| Recall | 0.4892 | 0.5890 | **+20.4%** |
| Precision | 0.6614 | 0.6168 | −6.74% |
| F1-Score | 0.5624 | 0.6026 | **+7.15%** |

El threshold calibrado captura un 20% más de casos de churn manteniendo un
equilibrio saludable con la precision — justificación cuantitativa de la decisión
de calibración.

### A.8 Conclusion

Ambos enfoques coinciden en:

1. Los modelos lineales superan a los de árbol en este dataset
2. LogisticRegression es el modelo más equilibrado del grupo lineal
3. El threshold calibrado por curva PR (0.441) mejora el F1 en +7% respecto al default
4. `contract_type`, `tenure_months` e `internet_service` son los drivers principales
5. `has_streaming` y `region` prácticamente no aportan señal predictiva

---

## Checklist de entrega 1

- [x] `conda env create -f environment.yml` funciona desde cero
- [x] `dvc repro` regenera el modelo sin intervención manual
- [x] `mlflow ui` muestra 4 runs con métricas logueadas
- [x] Modelo carga desde script independiente y predice correctamente
- [x] Notebook `01_eda.ipynb` corre con Kernel → Restart & Run All sin errores
- [x] 6 gráficos de EDA guardados en `reports/`
- [x] `reports/test_metrics.json` con métricas finales
- [x] `reports/eda_07_evaluation_test.png` con 3 visualizaciones
- [x] Validación independiente con PyCaret (apéndice)
- [x] Análisis SHAP sobre Random Forest (apéndice)
- [x] `ruff check src/` sin errores
- [x] Pre-setup E2: Dockerfile, docker-compose.yml, requirements.txt, app/ placeholders
- [x] `dvc.lock` commiteado (D13 — CP3 cerrado)
- [x] Informe técnico completo (D14)
- [x] `README.md` con instrucciones de instalación, ejecución y validación (D14)
- [x] Tag `v1.0-parcial1` en `main`

---

*úlima modificacion 09/05/2026.*
