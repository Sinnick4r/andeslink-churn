[![CI](https://github.com/Sinnick4r/andeslink-churn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Sinnick4r/andeslink-churn/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
![Version](https://img.shields.io/badge/version-1.0-green)
[![DVC Data](https://img.shields.io/badge/DVC-945DD6?style=flat&logo=dvc&logoColor=white)](https://dvc.org)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)


### Trabajo Practico: Predicción de Churn

Proyecto de **MLOps end-to-end** para el Laboratorio de Minería de Datos.
Resuelve un caso de clasificación binaria de abandono de clientes (churn) para
la empresa **AndesLink Servicios Digitales S.A.**, cubriendo el ciclo completo:
entrenamiento reproducible, despliegue como API + GUI containerizada, y
monitoreo técnico y de datos.

**Estado actual:** Entrega 1 (Entrenamiento) y Entrega 2 (Despliegue) cerradas.
Entrega 3 (Monitoreo) en desarrollo.

---

## Información del proyecto

- **Materia:** Laboratorio de Minería de Datos
- **Institución:** ISTEA
- **Profesor:** Diego Mosquera
- **Alumno:** Emilio Gomez Lencina
- **Modalidad:** trabajo individual
---

## Stack

| Capa | Herramientas |
|------|--------------|
| Entrenamiento | Python 3.11, scikit-learn 1.5.2, pandas 2.2, numpy 1.26 |
| Tracking y experimentos | MLflow 2.17 (file backend en `mlruns/`) |
| Pipeline y versionado | DVC 3.55 (remote público en Backblaze B2) |
| API de inferencia | FastAPI 0.115, Pydantic 2.9, uvicorn 0.32, loguru 0.7 |
| GUI | Streamlit 1.39, httpx 0.27 |
| Despliegue | Docker + Docker Compose v2 |
| Calidad de código | ruff 0.6, pytest 8.3 |

---

## Uso

### Levantar el sistema (E1 + E2)

```bash
git clone https://github.com/Sinnick4r/andeslink-churn.git
cd andeslink-churn
dvc pull
docker compose up -d --build
```

A los ~30 segundos el stack queda accesible en:

- **GUI**: http://localhost:8501
- **API**: http://localhost:8000 (endpoints `/health` y `/predict`)
### Reentrenar el modelo (opcional)

```bash
conda env create -f environment.yml
conda activate andeslink-churn
dvc repro
```

---

## Estructura

`app/` contiene la API FastAPI con sus contratos Pydantic. `gui/` contiene la
GUI Streamlit. `src/` tiene el código de training compartido con la API (feature
engineering, anti training-serving skew). `tests/` cubre API y GUI con 18 tests.
Los Dockerfiles separan API y GUI en imágenes independientes orquestadas por
`docker-compose.yml`. Toda la documentación técnica vive en `reports/`.

---

## Documentación técnica

| Documento | Contenido |
|-----------|-----------|
| [`reports/informe_e1.md`](reports/informe_e1.md) | EDA, comparación de modelos, calibración del threshold, decisiones de E1 |
| [`reports/informe_e2.md`](reports/informe_e2.md) | Arquitectura del despliegue, contrato de la API, decisiones de E2 |
| [`notebooks/01_EDA_churn_dataset.ipynb`](notebooks/01_EDA_churn_dataset.ipynb) | EDA detallado del dataset |
| [`notebooks/02_Validacion_pycaret.ipynb`](notebooks/02_Validacion_pycaret.ipynb) | Validación cruzada del modelo con PyCaret |
| [`notebooks/03_Script_prediccion_churn.ipynb`](notebooks/03_Script_prediccion_churn.ipynb) | Validación del modelo serializado |

---

## Resumen de decisiones técnicas

**E1 (Entrenamiento)**: LogisticRegression seleccionado por regla de tolerancia
F1 (RF marginalmente superior por 0.0013, dentro del ruido estadístico).
Threshold calibrado a 0.441444 por curva precision-recall. Features derivadas
`charges_per_month` y `tickets_per_year` compartidas entre training y serving.
Split 64/16/20 estratificado con `random_state=42`. Detalle completo en
[`informe_e1.md`](reports/informe_e1.md).

**E2 (Despliegue)**: dos containers (API + GUI) en red interna de Docker,
comunicación por DNS interno. Pydantic con `strict=True` y `Literal` en español
como Capa 1 de defensa de categorías, combinado con `handle_unknown="ignore"`
del OneHotEncoder como Capa 2. Threshold inyectado por env var. Hardening
uniforme (multi-stage, usuario no-root sin shell, filesystem read-only).
Trazabilidad por SHA-256 del modelo y `request_id` UUID4 por inferencia.
Detalle completo en [`informe_e2.md`](reports/informe_e2.md).

---

*Última actualización: 09/06/2026*
