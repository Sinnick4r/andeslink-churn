# AndesLink — Predicción de Churn

Proyecto de **MLOps end-to-end** para el Laboratorio de Minería de Dato. 
Resuelve un caso de clasificación binaria de
abandono de clientes (churn) para la empresa **AndesLink Servicios
Digitales S.A.**, cubriendo el ciclo completo: entrenamiento reproducible,
despliegue como API + GUI, y monitoreo técnico y de datos.

**Estado actual:** Entrega 1 (Entrenamiento) cerrada. Enrega 2 (Despliegue) y Entrega 3
(Monitoreo) en desarrollo.

---

## Informacion del proyecto

- **Materia:** Laboratorio de Minería de Datos
- **Institución:** ISTEA
- **Profesor:** Diego Mosquera
- **Fecha de entrega E1:** 12/05/2026
- **Modalidad:** trabajo individual

---

## Resultados de la Entrega 1

| Métrica | Valor (test set, n=1.000) |
|---------|---------------------------|
| Modelo seleccionado | LogisticRegression |
| F1-score | 0.6014 |
| ROC-AUC | 0.7561 |
| Recall (churn) | 0.7853 |
| Threshold calibrado | 0.441 |

El detalle completo de decisiones, EDA, comparación de modelos, etc. esta en ->
[`reports/informe_e1.md`](reports/informe_e1.md).

Se puso por separado para facilitar la lectura.

---

## Stack y prerequisitos

### Software requerido

- **Python 3.11** (vía Anaconda / Miniconda)
- **Git** ≥ 2.30
- **Sistema operativo:** desarrollado en Windows 10. Linux y macOS son compatibles
  con ajustes mínimos en los comandos PowerShell mostrados abajo.

### Stack del proyecto (versiones pinneadas en `environment.yml`)

| Herramienta | Versión | Uso |
|-------------|---------|-----|
| Python | 3.11.x | Lenguaje base |
| scikit-learn | 1.5.2 | Entrenamiento (versión exacta - el `.joblib` es version-sensitive) |
| pandas | 2.2.x | Preparacion de datos |
| numpy | 1.26.x | Pinneado para evitar conflicto con sklearn en numpy 2.x |
| MLflow | 2.17.x | Tracking de experimentos (file backend en `mlruns/`) |
| DVC | 3.55.x | Pipeline + data versioning (storage local) |
| matplotlib / seaborn | 3.9.x / 0.13.x | Visualizaciones del EDA |
| ruff | 0.6.x | Linter + formatter |

---

## Setup desde cero 

### Paso 1 - Clonar el repo

```powershell
git clone <URL_DEL_REPO> andeslink-churn
cd andeslink-churn
```

### Paso 2 - Crear el entorno conda

```powershell
conda env create -f environment.yml
conda activate andeslink-churn
```

**Verificacion de versiones:**

```powershell
python -c "import sklearn; print('sklearn:', sklearn.__version__)"  # 1.5.2
python -c "import numpy; print('numpy:', numpy.__version__)"        # 1.26.x
python -c "import mlflow; print('mlflow:', mlflow.__version__)"     # 2.17.x
```

### Paso 3 - Colocar el CSV en `data/raw/`

> **Observacion:** El proyecto usa DVC con storage local
> (sin remote configurado), por lo que el CSV se entrega por separado.

Copiar el archivo `churn_sintetico.csv`  a la carpeta `data/raw/` y verificar que el hash coincide con el trackeado por DVC:

```powershell
dvc status data/raw/churn_sintetico.csv.dvc
```

Si todo está bien, debe imprimir `Data and pipelines are up to date.`

### Paso 4 - Reproducir el pipeline completo

```powershell
dvc repro
```

Este comando ejecuta los 3 stages declarados en `dvc.yaml`:

```
prepare  →  train  →  evaluate
```

Genera:

- `models/churn_model_v1.joblib` (modelo serializado)
- `reports/train_metrics.json` (metricas de los 3 modelos en validacion)
- `reports/test_metrics.json` (metricas finales en test set)
- `reports/eda_07_evaluation_test.png` (3 visualizaciones de evaluacion)

---

## Validacion de la repro

### Verificacion 1 - el modelo carga y predice

```powershell
python -c "import joblib; m = joblib.load('models/churn_model_v1.joblib'); print('Pipeline cargado OK:', type(m).__name__)"
```

Salida esperada: `Pipeline cargado OK: Pipeline`

### Verificacion 2 - métricas idénticas en test set

```powershell
python -c "import json; print(json.dumps(json.load(open('reports/test_metrics.json')), indent=2))"
```

Salida esperada (los valores deben coincidir exactamente - `random_state=42`):

```json
{
  "f1_score": 0.6014,
  "roc_auc": 0.7561,
  "pr_auc": 0.6155,
  "precision_class_1": 0.4872,
  "recall_class_1": 0.7853,
  "threshold": 0.441,
  ...
}
```

### Verificacion 3 - `dvc repro` desde cero es idempotente

Borrando los registros y re-ejecutando, los resultados deben ser bit-exactos:

**PowerShell:**
```powershell
Remove-Item models/churn_model_v1.joblib -ErrorAction SilentlyContinue
Remove-Item reports/test_metrics.json -ErrorAction SilentlyContinue
dvc repro
```

**Bash:**
```bash
rm -f models/churn_model_v1.joblib reports/test_metrics.json
dvc repro
```

### Verificacion 4 - explorar los experimentos en MLflow UI

```powershell
mlflow ui --port 5000
```

Abrir [http://localhost:5000](http://localhost:5000) en el navegador. El experimento
`andeslink-churn-e1` contiene 4 runs:

| Run | F1 | ROC-AUC | Threshold |
|-----|-----|---------|-----------|
| LogisticRegression | 0.6216 | 0.7544 | 0.441 |
| RandomForest | 0.6209 | 0.7503 | 0.410 |
| HistGradientBoosting | 0.6059 | 0.7461 | 0.305 |
| GANADOR_LogisticRegression | (modelo serializado) | — | 0.441 |

---

## Notebooks

### `notebooks/EDA_churn_dataset.ipynb` — Análisis exploratorio

Cubre 9 secciones: distribución del target, descriptivas, boxplots por clase,
categóricas vs churn, matriz de correlación, multicolinealidad de `total_charges`,
correlaciones con el target, y conclusiones. Genera 6 PNGs en `reports/`.

```powershell
jupyter lab notebooks/01_EDA_churn_dataset.ipynb
```

Para ejecutar todo: `Kernel → Restart & Run All`.

### `notebooks/Validacion_pycaret.ipynb` - Calidacion extra de elección del mnodelo con PyCaret

Validación independiente del modelo elegido usando PyCaret (`compare_models`). 
**No es parte delpipeline DVC** - corre por separadoy se creo solo para validar el modelo elegido.

> **Nota de instalacion:** PyCaret no está en `environment.yml` por su tamaño
> de dependencias. Si se quiere re-ejecutar ese notebook:

 ```powershell
 pip install pycaret==3.3.2
 ```
## Validacion del modelo serializado

Para verificar que el modelo carga y predice correctamente desde un script
independiente (sin reentrenar):

```powershell
  jupyter lab notebooks/03_Script_prediccion_churn.ipynb
```

El notebook valida hash SHA-256 del artefacto, carga el pipeline,
y permite probar predicciones con valores ingresados

---

## Estructura del repositorio

```
andeslink-churn/
├── data/
│   ├── raw/churn_sintetico.csv.dvc   ← CSV trackeado por DVC
│   ├── processed/.gitkeep
│   └── current/.gitkeep
├── notebooks/
│   ├── 01_EDA_churn_dataset.ipynb       ← EDA completo
│   ├── 02_Validacion_Pycaret.ipynb      ← Evaluacion extra con PyCaret
│   └── 03_Script_prediccion_churn.ipynb ← Notebook de prueba del modelo
├── src/
│   ├── features.py                   ← add_derived_features, build_preprocessor
│   ├── data.py                       ← load_raw_data, split_data
│   ├── train.py                      ← run_experiment, MLflow tracking
│   └── evaluate.py                   ← evaluate_on_test, plot_evaluation
├── app/                              ← placeholders E2 (FastAPI)
├── gui/                              ← placeholder E2 (Streamlit)
├── models/
│   └── churn_model_v1.joblib         ← modelo serializado (DVC tracked)
├── reports/
│   ├── eda_01...07.png               ← 7 graficos del EDA + evaluación
│   ├── train_metrics.json            ← comparativa de los 3 modelos
│   ├── test_metrics.json             ← métricas finales en test
│   └── informe_e1.md                 ← informe tecnico completo
├── dvc.yaml                          ← pipeline (3 stages)
├── dvc.lock                          ← firma de reproducibilidad
├── params.yaml                       ← parametros trackeados
├── environment.yml                   ← entorno conda portable
├── environment.lock.yml              ← versiones exactas (Windows)
├── pyproject.toml                    ← configuración ruff
└── README.md                         ← este archivo
```

---

## Ejecutar entrenamiento sin DVC (uso interactivo)

Si se quierencorrer los módulos manualmente sin pasar por `dvc repro`:

```powershell
# Entrenar los 3 modelos y serializar el ganador
python -m src.train

# Evaluar el modelo final sobre el test set
python -m src.evaluate
```

Ambos scripts loguean en MLflow (`mlruns/`) y generan los artefactos en
`models/` y `reports/`.

---

## Lint y calidad del código

El proyecto usa `ruff` configurado en `pyproject.toml`. Para verificar antes de
cada commit:

```powershell
ruff check src/
```

Debe imprimir `All checks passed!` sin warnings.

---

## Resumen de decisiones tecnicas 
Las decisiones cerradas durante E1 están documentadas en detalle en
[`reports/informe_e1.md`](reports/informe_e1.md). Resumen:

- **Modelo:** LogisticRegression (le gana a RF y HistGBM con diferencia <0.02 F1)
- **Threshold:** 0.441 calibrado por curva precision-recall (no se uso el default de 0.5)
- **Features descartadas:** `total_charges` (redundancia estructural con
  `tenure_months × monthly_charge`)
- **Features derivadas:** `charges_per_month`, `tickets_per_year`
- **Split:** 64/16/20 double split estratificado, `random_state=42`
- **Metrica principal:** F1 con prioridad de modelo mas simple (desbalance moderado, costo asimétrico FN > FP)

---


*README última actualización: 09/05/2026.*
