[![CI](https://github.com/Sinnick4r/andeslink-churn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Sinnick4r/andeslink-churn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# AndesLink Churn MLOps

Proyecto de MLOps para predecir el abandono de clientes de AndesLink Servicios Digitales S.A., una empresa simulada de servicios por suscripción

La solución cubre el ciclo completo de Machine Learning:

- preparación y versionado de datos
- entrenamiento y evaluación del modelo
- registro de experimentos
- API de inferencia
- interfaz gráfica
- despliegue con contenedores
- pruebas automáticas
- monitoreo técnico y detección de drift

## Resultados del modelo

Se compararon regresión logística, Random Forest e HistGradientBoosting. La regresión logística fue seleccionada por ofrecer un rendimiento similar al mejor modelo con menor complejidad.

| Métrica de prueba | Resultado |
|---|---:|
| F1 | 0.6020 |
| Recall de churn | 0.7853 |
| ROC-AUC | 0.7561 |
| PR-AUC | 0.6155 |
| Umbral de decisión | 0.441444 |

El modelo detecta aproximadamente 8 de cada 10 clientes que abandonan el servicio en el conjunto de prueba.

## Arquitectura

El sistema está formado por los siguientes componentes:

| Componente | Tecnología | Función |
|---|---|---|
| Entrenamiento | scikit-learn | Preparación, entrenamiento y evaluación |
| Trazabilidad | DVC y MLflow | Versionado del pipeline y registro de experimentos |
| API | FastAPI | Exposición del modelo mediante `/predict` |
| Interfaz | Streamlit | Formulario para realizar predicciones |
| Despliegue | Docker Compose | Ejecución local de los servicios |
| Métricas | Prometheus | Recolección y evaluación de métricas |
| Dashboard | Grafana | Visualización del estado del sistema |
| Drift | Evidently | Comparación entre datos de referencia e inferencias |
| Calidad | pytest y Ruff | Pruebas automáticas y análisis estático |

## Ejecución

### Requisitos

- Git
- Docker con Docker Compose v2
- DVC
- Conda, solo para entrenamiento, pruebas y scripts locales

### Descargar el proyecto

```bash
git clone https://github.com/Sinnick4r/andeslink-churn.git
cd andeslink-churn
```

El modelo serializado ya esta incluido en el repositorio y se baja al clonar.

### Levantar los servicios

```bash
docker compose up -d --build
```

Una vez iniciados los contenedores:

| Servicio | Dirección |
|---|---|
| GUI | http://localhost:8501 |
| API | http://localhost:8000 |
| Estado de la API | http://localhost:8000/health |
| Métricas de la API | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

El dashboard de Grafana se carga automaticamente y permite acceso de solo lectura sin iniciar sesión.

Para verificar el estado de los contenedores:

```bash
docker compose ps
```

Para detener el sistema:

```bash
docker compose down
```

## Monitoreo

La API expone metricas de disponibilidad, solicitudes, errores, latencia y comportamiento de las predicciones.

Prometheus recolecta estas metricas cada 15 segundos. 

Grafana presenta los principales indicadores mediante un dashboard configurado automaticamente

### Generar tráfico de prueba

```bash
python scripts/generate_traffic.py -n 200 --delay 0.25
```

El script envia solicitudes válidas y algunos casos invalidos para generar actividad observable en Prometheus y Grafana.

### Generar tráfico con drift

```bash
python scripts/generate_drift.py -n 150 --delay 0.2
```

Este escenario genera clientes con distribuciones diferentes de las utilizadas durante el entrenamiento

### Crear el reporte de Evidently

```bash
python scripts/drift_report.py
```

El resultado se guarda en:

```text
reports/drift_report.html
```

El reporte compara las inferencias registradas con el dataset de referencia y muestra las variables que presentan cambios relevantes

## Reproducir el entrenamiento

Crear y activar el entorno:

```bash
conda env create -f environment.yml
conda activate andeslink-churn
dvc pull
```

Ejecutar el pipeline completo:

```bash
dvc repro
```

El pipeline contiene tres etapas:

```text
prepare -> train -> evaluate
```

Las métricas generadas pueden consultarse con:

```bash
dvc metrics show
```

Los experimentos locales de MLflow pueden visualizarse mediante:

```bash
mlflow ui
```

## Pruebas y calidad

Con el entorno Conda activo:

```bash
ruff check .
pytest
```

La integración continua ejecuta estas verificaciones automáticamente mediante GitHub Actions.

## Estructura del repositorio

```text
app/                    API FastAPI y métricas
gui/                    Interfaz Streamlit
src/                    Preparación, entrenamiento y evaluación
tests/                  Pruebas de API y GUI
scripts/                Generadores de tráfico y reporte de drift
monitoring/
  prometheus/           Configuración y reglas PromQL
  grafana/              Provisioning y dashboard
data/                   Datos administrados con DVC
models/                 Modelo serializado
notebooks/              Análisis y validaciones
reports/                Métricas, resultados e informe final
docker-compose.yml      Orquestación de servicios
dvc.yaml                Pipeline reproducible
params.yaml             Parámetros de entrenamiento
```

## Documentación

- [Informe técnico final](reports/informe_final.md)
- [Notebook de análisis exploratorio](notebooks/01_EDA_churn_dataset.ipynb)
- [Notebook de validación](notebooks/02_Validacion_pycaret.ipynb)
- [Notebook de prueba del modelo](notebooks/03_Script_prediccion_churn.ipynb)

## Solución de problemas

### Algún puerto ya está siendo utilizado

Verificar los puertos `8000`, `8501`, `9090` y `3000`, o detener el proceso que los esté ocupando antes de iniciar Docker Compose.

### Evidently informa que no hay datos suficientes

Primero deben generarse al menos 40 inferencias:

```bash
python scripts/generate_traffic.py -n 100
python scripts/generate_drift.py -n 100
```

Después puede ejecutarse nuevamente:

```bash
python scripts/drift_report.py
```

### Un contenedor no inicia correctamente

```bash
docker compose ps
docker compose logs api
docker compose logs gui
docker compose logs prometheus
docker compose logs grafana
```

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
 
*actualización: 12/07/2026*
