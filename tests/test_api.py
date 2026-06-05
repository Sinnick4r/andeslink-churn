"""Tests de aceptación de la API
Cobertura mínima requerida: happy path, input inválido (422), campo extra (422),
categoría desconocida (422), método incorrecto (405) y health check (200).
Los tests no dependen de orden de ejecución ni de estado global; usan las
fixtures ``client`` y ``valid_payload_factory`` definidas en ``conftest.py``.
"""

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

PayloadFactory = Callable[..., dict[str, Any]]


def test_predict_valid_input(client: TestClient, valid_payload_factory: PayloadFactory) -> None:
    response = client.post("/predict", json=valid_payload_factory())
    assert response.status_code == 200
    data = response.json()
    assert data["churn"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "v2"
    assert 0.0 <= data["threshold"] <= 1.0
    assert data["request_id"]


def test_predict_propagates_request_id(
    client: TestClient, valid_payload_factory: PayloadFactory
) -> None:
    request_id = "test-trace-id-001"
    response = client.post(
        "/predict", json=valid_payload_factory(), headers={"X-Request-ID": request_id}
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id


def test_predict_invalid_age(client: TestClient, valid_payload_factory: PayloadFactory) -> None:
    response = client.post("/predict", json=valid_payload_factory(customer_age=150))
    assert response.status_code == 422
    # El cliente recibe un mensaje genérico; el detalle queda en el log
    assert response.json() == {"detail": "Invalid input parameters"}


def test_predict_extra_field_rejected(
    client: TestClient, valid_payload_factory: PayloadFactory
) -> None:
    response = client.post("/predict", json=valid_payload_factory(malicious="x"))
    assert response.status_code == 422


def test_predict_unknown_category(
    client: TestClient, valid_payload_factory: PayloadFactory
) -> None:
    # `payment_method` no acepta "bitcoin"
    response = client.post("/predict", json=valid_payload_factory(payment_method="bitcoin"))
    assert response.status_code == 422


def test_predict_strict_type_rejected(
    client: TestClient, valid_payload_factory: PayloadFactory
) -> None:
    # strict=True: un string no se coerciona a int.
    response = client.post("/predict", json=valid_payload_factory(tenure_months="24"))
    assert response.status_code == 422


def test_predict_wrong_method(client: TestClient) -> None:
    assert client.get("/predict").status_code == 405


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "v2"
    assert body["sklearn_version"]
