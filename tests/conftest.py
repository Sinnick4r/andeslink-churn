"""Fixtures compartidas para los tests de la API .

- ``client``: ``TestClient`` con la app REAL. Al entrar al context manager, el
  lifespan carga el artefacto real del modelo, por lo que la fixture funciona
  además como smoke test integrado de carga + predicción. No se mockea el modelo
  en los tests de aceptación: si el ``.joblib`` o la versión de
  scikit-learn no están bien, los tests fallan ruidosamente.
- ``valid_payload_factory``: genera payloads válidos con overrides parciales,
  evitando repetir el dict de 15 campos en cada test.

Requisitos para correr esto: el entorno (conda ``andeslink-churn``) con
scikit-learn==1.5.2 y el artefacto ``models/churn_model_v2.joblib`` disponible
(``dvc pull`` si hace falta).
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    # Cliente con la app real; el lifespan carga el modelo una vez por sesión
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload_factory() -> Callable[..., dict[str, Any]]:
    # Devuelve una factory que arma payloads válidos con overrides parciales
    base: dict[str, Any] = {
        "tenure_months": 24,
        "monthly_charge": 65.5,
        "total_charges": 1572.0,
        "support_tickets": 2,
        "late_payments": 1,
        "avg_monthly_usage_gb": 120.0,
        "contract_type": "anual",
        "payment_method": "credito",
        "internet_service": "fibra",
        "region": "centro",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 2,
        "customer_age": 35,
        "is_promo": 0,
    }

    def _make(**overrides: Any) -> dict[str, Any]:
        return {**base, **overrides}

    return _make
