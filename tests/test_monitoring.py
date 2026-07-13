# tests basicos de las metricas expuestas por la API

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

PayloadFactory = Callable[..., dict[str, Any]]


def _metric_value(text: str, name: str, labels: dict[str, str]) -> float:
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            labels_match = all(sample.labels.get(key) == value for key, value in labels.items())
            if sample.name == name and labels_match:
                return float(sample.value)
    return 0.0


def test_metrics_endpoint_exposes_custom_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    assert "churn_api_predictions_total" in body
    assert "churn_api_inference_duration_seconds" in body
    assert "churn_api_prediction_probability" in body
    assert "churn_api_model_info" in body


def test_prediction_increments_class_counter(
    client: TestClient,
    valid_payload_factory: PayloadFactory,
) -> None:
    before = client.get("/metrics").text
    response = client.post("/predict", json=valid_payload_factory())
    assert response.status_code == 200

    predicted_class = str(response.json()["churn"])
    labels = {"predicted_class": predicted_class, "model_version": "v2"}
    previous = _metric_value(before, "churn_api_predictions_total", labels)
    current = _metric_value(client.get("/metrics").text, "churn_api_predictions_total", labels)

    assert current == previous + 1
