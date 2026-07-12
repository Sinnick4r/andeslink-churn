# Monitoreo (Entrega 3)

Documentación operativa del sistema de monitoreo de AndesLink Churn. Cubre las
dos capas que vigilan el servicio en producción y el procedimiento para
interpretar lo que reportan y actuar en consecuencia.

El monitoreo está separado en dos planos con preguntas distintas:

- **Salud del servicio** (Prometheus + Grafana): responde "¿el servicio está
  funcionando bien ahora?". Métricas agregadas en tiempo real (disponibilidad,
  latencia, errores, volumen de predicciones).
- **Drift de datos y modelo** (Evidently): responde "¿los datos que llegan
  siguen pareciéndose a los del entrenamiento?". Análisis batch, on-demand,
  sobre el registro de inferencias reales.

```
                     /predict
   cliente  ───────────────────────►  API (FastAPI)
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  │                                              │
          /metrics (8000)                            registro JSONL (bind mount)
                  │                                              │
       Prometheus (9090) scrape 15s                  monitoring/data/inferences.jsonl
                  │                                              │
         Grafana (3000) dashboards                   Evidently (script on-demand)
                  │                                              │
        salud del servicio en vivo                  monitoring/reports/drift_report.html
```

---

## Componentes

| Componente | Tecnología | Puerto | Responsabilidad |
|------------|------------|--------|-----------------|
| API | FastAPI + instrumentator | 8000 | Expone `/metrics` y registra cada inferencia en JSONL |
| Prometheus | prom/prometheus v2.54.0 | 9090 | Scrapea `/metrics` cada 15s y almacena las series |
| Grafana | grafana/grafana 11.2.0 | 3000 | Dashboard provisionado como código sobre Prometheus |
| Evidently | evidently 0.4.33 (conda) | n/a | Genera el reporte de drift comparando referencia vs producción |

Evidently no corre como container: es un script que se ejecuta a pedido desde el
entorno conda del proyecto (que ya lo incluye), porque su insumo es dinámico y no
forma parte del camino crítico de inferencia.

---

## Capa 1: salud del servicio (Prometheus + Grafana)

### Métricas expuestas

La API publica en `/metrics` dos familias de métricas. Las automáticas vienen
del instrumentator (toda request HTTP) y las custom miden el comportamiento del
modelo:

| Métrica | Tipo | Qué mide |
|---------|------|----------|
| `http_requests_total` | Counter | Requests por handler, método y status (el 422 queda desagregado) |
| `http_request_duration_seconds` | Histogram | Latencia HTTP por handler, base de los percentiles |
| `churn_api_predictions_total` | Counter | Predicciones acumuladas, por clase predicha y versión del modelo |
| `churn_api_inference_duration_seconds` | Histogram | Latencia de la inferencia pura (feature engineering + `predict_proba`) |
| `churn_api_prediction_probability` | Histogram | Distribución de la probabilidad de churn, para observar el shift de scores |
| `churn_api_model_info` | Gauge | Metadata fija en 1, transporta versión del modelo y de scikit-learn en los labels |

La cardinalidad está acotada a propósito: el `request_id` nunca se usa como
label, solo viaja en los logs y en el registro de inferencias.

### Dashboard

Grafana levanta con el dashboard "AndesLink Churn - API Overview" ya
provisionado (no hay que importarlo a mano). Se accede en
http://localhost:3000 con acceso anónimo de solo lectura habilitado. Los ocho
paneles son:

| Panel | Lectura |
|-------|---------|
| Disponibilidad de la API | `up` del target; en rojo si Prometheus no scrapea |
| Requests por segundo | Volumen de tráfico entrante |
| Tasa de error (4xx + 5xx) | Proporción de respuestas con error |
| Proporción de churn predicho | Balance entre clase 0 y 1 a lo largo del tiempo |
| Requests por endpoint y status | Desglose fino, incluye los 422 de validación |
| Latencia HTTP (p50 / p95 / p99) | Percentiles, el p99 anticipa saturación |
| Predicciones por clase (rate) | Ritmo de predicciones positivas y negativas |
| Distribución de la probabilidad de churn | Histograma de scores, anticipa prediction drift en vivo |

---

## Capa 2: drift de datos y modelo (Evidently)

### Registro de inferencias

Cada llamada a `/predict` escribe una línea en
`monitoring/data/inferences.jsonl` con las 15 features de entrada más la
predicción (clase, probabilidad, threshold, versión, `request_id` y timestamp
UTC). Ese archivo es la "foto" de lo que el modelo ve en producción y es el
insumo del análisis de drift.

El registro es **best-effort**: si la escritura falla (permisos, disco), se
loguea un warning pero la predicción se devuelve igual. El monitoreo nunca
degrada el camino crítico. La persistencia usa un bind mount dedicado
(`./monitoring/data`) porque el container corre con filesystem de solo lectura,
y montarlo en el host permite que Evidently, que corre fuera del container, lo
lea directamente.

El log operacional de la API y este registro son dos planos distintos: el log
nunca incluye features (solo metadata de operación), el registro sí las guarda.
El `request_id` es el puente de trazabilidad entre ambos.

### Referencia y actual

- **Referencia**: el split de entrenamiento del dataset, en sus columnas crudas
  (las 15 que envía el cliente), obtenido con el mismo split determinístico que
  vio el modelo. Aquí `churn` es el valor real.
- **Actual**: las inferencias registradas en producción. Aquí `churn` es la
  clase predicha, porque todavía no hay ground truth.

Comparar la distribución del churn real (referencia) contra la del churn
predicho (actual) es la forma habitual de vigilar prediction drift cuando no hay
etiquetas en producción.

---

## Correr el flujo end-to-end

El entorno conda del proyecto (`andeslink-churn`) ya incluye Evidently y httpx,
así que el flujo se corre desde ahí, sin entorno aparte.

```bash
# 1. levantar el stack (la primera vez o tras cambiar codigo: con build)
make up-build

# 2. activar el entorno del proyecto
conda activate andeslink-churn

# 3. poblar metricas y registro (la API tiene que estar arriba)
make traffic

# 4. generar el reporte de drift
make drift
```

Para la demo en vivo, `make demo` encadena una secuencia de escenarios
(baseline, burst, churn-spike, drift, errores, recuperación) que mueve los
paneles y deja ver las señales una tras otra.

Al terminar quedan disponibles:

- Grafana con datos reales en http://localhost:3000
- El reporte en `monitoring/reports/drift_report.html`

`generate_traffic.py` tiene tres modos: batch (`--healthy`/`--drift`) para
poblar el registro del reporte, continuo (`--loop`) para tráfico de fondo, y
escenarios (`--scenario`) para tramos controlados como los de la demo.

---

## Leer el reporte de drift

El reporte trabaja en dos niveles y conviene leerlos juntos:

- **Nivel dataset**: un flag agregado que se enciende cuando la proporción de
  columnas con drift supera un umbral (0.5 por defecto). Es un resumen grueso.
- **Nivel columna**: por cada feature, un test estadístico, un score y un flag
  de drift detectado. Aquí está la información accionable.

Los tests que aplica Evidently dependen del tipo de variable: distancia de
Wasserstein normalizada para las numéricas y distancia de Jensen-Shannon para
las categóricas. El score no es un porcentaje: un valor por encima de 1 en
Wasserstein normalizada indica que la media se corrió más de un desvío estándar
de la referencia.

La regla práctica: **no decidir por el flag agregado**. El nivel dataset puede
quedar apagado mientras hay columnas con drift fuerte. La decisión se toma
mirando qué features driftearon y cuánto.

---

## Interpretación del reporte de ejemplo

El tráfico de prueba simula una población hipotética de clientes más nuevos y
con más fricción: menos antigüedad, más tickets de soporte, más pagos
atrasados, cargo mensual más alto y contrato mensual. Sobre 500 inferencias
contra una referencia de entrenamiento, el reporte detecta drift en seis de
dieciséis columnas:

| Columna | Score | Lectura |
|---------|-------|---------|
| `late_payments` | 2.52 | Más pagos atrasados que en la referencia |
| `support_tickets` | 2.01 | Más tickets de soporte, la señal más fuerte de fricción |
| `monthly_charge` | 0.59 | Cargo mensual corrido hacia arriba |
| `tenure_months` | 0.50 | Población más nueva |
| `churn` (predicho) | 0.26 | El modelo predice más churn como consecuencia (prediction drift) |
| `contract_type` | 0.12 | La mezcla se corre hacia el contrato mensual |

Las otras diez features (edad, total de cargos, región, método de pago, etc.)
quedan en Not Detected con scores menores a 0.05. El sistema discrimina señal de
ruido: drifta exactamente lo que cambió y no genera falsos positivos.

El flag a nivel dataset queda apagado (seis de dieciséis es 0.375, por debajo
del umbral de 0.5), pero el drift por columna es evidente. Es el caso de libro
de por qué hay que mirar los dos niveles.

El churn predicho corriéndose es la lectura clave: ante clientes con ese perfil
deteriorado, el modelo predice más abandono. En un escenario real esto gatilla
la investigación de la causa (¿campaña nueva?, ¿segmento distinto?, ¿problema en
los datos de origen?) antes de tocar el modelo.

---

## Tabla señal a acción correctiva

Organizada por las tres vistas del sistema. Las acciones son proporcionadas: la
mayoría empieza por confirmar antes de intervenir.

| Vista | Señal observada | Diagnóstico probable | Acción correctiva |
|-------|-----------------|----------------------|-------------------|
| Servicio | `up{job="churn-api"}` en 0, disponibilidad en rojo | Container caído o healthcheck fallando; el modelo no cargó en el arranque | Revisar `docker logs` y `/health`; si el modelo no carga, verificar versión de sklearn y `dvc pull`; reiniciar o hacer rollback de la imagen |
| Servicio | Tasa de error 5xx en alza en `/predict` | Excepción no controlada en la inferencia, dependencia rota o payload que rompe el pipeline | Rastrear por `request_id` en los logs, reproducir el payload; si es una regresión, rollback a la imagen estable anterior |
| Servicio | Pico de respuestas 422 | El cliente manda payloads inválidos: cambió el contrato del consumidor o hay un bug aguas arriba | Revisar la integración del cliente; si el schema cambió de forma legítima, versionar el endpoint en vez de relajar la validación |
| Servicio | p95 o p99 de latencia sostenidamente altos | Saturación de CPU o memoria (límites de 512m y 1 cpu), contención | Revisar el uso de recursos del container, subir límites o escalar horizontalmente |
| Comportamiento | La proporción de churn predicho se dispara respecto a su base | Posible data drift en la entrada o cambio real en la población | Correr el reporte de Evidently para confirmar; validar con negocio si es un cambio real o un dato roto |
| Comportamiento | El histograma de probabilidad se concentra cerca del threshold o se corre a un extremo | Señal temprana de prediction drift, el modelo está menos definido | Cruzar con el reporte de drift; si se confirma, evaluar recalibrar el threshold o reentrenar |
| Drift | Drift en features de comportamiento (`support_tickets`, `late_payments`, `tenure_months`) | Cambió el perfil de clientes que llegan: estacionalidad, campaña o segmento nuevo | Validar la causa con negocio; si el shift es real y sostenido, reentrenar con datos recientes y documentar el cambio |
| Drift | Prediction drift sin ground truth disponible | El modelo predice distinto, pero sin etiquetas reales no se sabe si acierta menos | Priorizar la obtención de labels reales (ventana de observación de churn); mientras tanto monitorear y preparar el reentrenamiento, sin tocar el threshold a ciegas |
| Drift | Data quality: aparecen nulos o conteos anómalos en el registro | Problema en el pipeline de datos aguas arriba o cambio de schema | Revisar la fuente de datos; correlacionar con picos de 422 si los hubo |
| Drift | `dataset_drift` global apagado pero hay columnas drifteadas | Pocas columnas superan el umbral agregado, pero las que driftearon son relevantes | No descartar por el flag global: revisar siempre el detalle por columna y decidir sobre esas features |

---

## Decisiones de diseño del monitoreo

**Dos planos de registro, no uno.** El log operacional de la API nunca guarda
features (solo metadata: clase, duración, versión). El registro de inferencias
sí las guarda, porque son la materia prima del análisis de drift. Mantenerlos
separados deja el log liviano y apto para acceso amplio, mientras el registro
detallado queda acotado al monitoreo. El `request_id` correlaciona ambos.

**Métricas agregadas y registro detallado se complementan.** Prometheus
responde la pregunta de salud en vivo con series agregadas y baratas. Evidently
responde la pregunta de drift con el detalle completo, pero en batch. El
histograma `churn_api_prediction_probability` es el puente: da un anticipo en
vivo del prediction drift que después Evidently confirma en frío.

**Prediction drift sin etiquetas.** En producción no hay ground truth inmediato
(el churn real se observa con semanas de demora). Comparar la distribución del
churn predicho contra la del churn real de entrenamiento es la práctica habitual
para detectar que algo cambió sin esperar las etiquetas, aceptando que es una
señal de alerta y no una medición de error.

**Registro best-effort.** Persistir una inferencia nunca puede tumbar la
predicción. La escritura va dentro de un `try/except` que loguea el fallo y
sigue. El monitoreo es importante, pero el camino crítico de inferencia tiene
prioridad.
