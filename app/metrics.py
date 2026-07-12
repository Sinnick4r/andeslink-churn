"""Metricas custom de Prometheus para la API

- conteo de predicciones desagregado por clase
- latencia de la inferencia pura, sin el overhead HTTP
- distribucion de las probabilidades predichas
- metadata del modelo cargado

"""

from typing import Final

from prometheus_client import Counter, Gauge, Histogram

# buckets de latencia de inferencia en segundos
# cubren desde 5ms hasta 1s; LogReg evalua en el orden de microsegundos
_DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

# buckets de probabilidad en el rango [0, 1] para observar el shift de scores
_PROBABILITY_BUCKETS: Final[tuple[float, ...]] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

# contador de predicciones desagregado por clase y version del modelo
predictions_total: Final[Counter] = Counter(
    "churn_api_predictions_total",
    "Total de predicciones realizadas",
    labelnames=("predicted_class", "model_version"),
)

# histograma de la duracion de la inferencia pura (feature engineering + predict_proba)
# separada de la latencia HTTP que ya mide el instrumentator
inference_duration_seconds: Final[Histogram] = Histogram(
    "churn_api_inference_duration_seconds",
    "Duracion de la inferencia en segundos",
    buckets=_DURATION_BUCKETS,
)

# histograma de la probabilidad de la clase positiva
# da un anticipo en vivo del prediction drift sin esperar al reporte batch
prediction_probability: Final[Histogram] = Histogram(
    "churn_api_prediction_probability",
    "Distribucion de la probabilidad de churn predicha",
    buckets=_PROBABILITY_BUCKETS,
)

# Gauge informativa: queda fija en 1 y transporta metadata en los labels
# patron info metric, util para correlacionar las series con la version desplegada
model_info: Final[Gauge] = Gauge(
    "churn_api_model_info",
    "Metadata del modelo cargado, el valor es siempre 1",
    labelnames=("model_version", "sklearn_version"),
)


def record_prediction(
    predicted_class: int,
    probability: float,
    duration_seconds: float,
    model_version: str,
) -> None:
    # Actualiza las metricas de una inferencia individual

    assert predicted_class in (0, 1), f"clase fuera de dominio: {predicted_class}"
    assert 0.0 <= probability <= 1.0, f"probabilidad fuera de [0,1]: {probability}"
    assert duration_seconds >= 0.0, f"duracion negativa: {duration_seconds}"

    predictions_total.labels(
        predicted_class=str(predicted_class),
        model_version=model_version,
    ).inc()
    inference_duration_seconds.observe(duration_seconds)
    prediction_probability.observe(probability)


def set_model_info(model_version: str, sklearn_version: str) -> None:
    # Fija la metrica informativa del modelo al startup de la API
    assert model_version, "model_version no puede ser vacio"
    assert sklearn_version, "sklearn_version no puede ser vacio"
    model_info.labels(
        model_version=model_version,
        sklearn_version=sklearn_version,
    ).set(1)
