"""Tests de las funciones puras de gui/app.py.

No requieren ``streamlit`` corriendo ni la API levantada: se usa
``httpx.MockTransport`` para simular el servidor. Solo se testean las
funciones puras (``check_api_health``, ``call_predict``, ``build_payload_from_form``);
el render de Streamlit se valida manualmente al levantar el stack.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest

# Stub minimo de streamlit para permitir el import del modulo gui.app
# en entornos donde streamlit no esta instalado (CI rapido por ejemplo)
if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    for _attr in (
        "set_page_config",
        "title",
        "caption",
        "error",
        "info",
        "stop",
        "success",
        "metric",
        "divider",
        "form",
        "form_submit_button",
        "subheader",
        "columns",
        "number_input",
        "selectbox",
        "spinner",
        "progress",
        "expander",
        "code",
        "warning",
    ):
        setattr(streamlit_stub, _attr, lambda *a, **kw: None)
    sys.modules["streamlit"] = streamlit_stub


from gui.streamlit_app import (  # noqa: E402
    DEFAULT_VALUES,
    build_payload_from_form,
    call_predict,
    check_api_health,
)


@pytest.fixture
def patch_httpx(monkeypatch: pytest.MonkeyPatch):
    #Permite reemplazar httpx.get o httpx.post con MockTransport

    def _patch(method: str, status: int, body: dict | None = None, raise_exc: type | None = None):
        def handler(request: httpx.Request) -> httpx.Response:
            if raise_exc is not None:
                raise raise_exc("simulated")
            return httpx.Response(status, json=body or {})

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        def fake(url: str, **kwargs: Any) -> httpx.Response:
            kwargs.pop("timeout", None)
            if method == "get":
                return client.get(url, **kwargs)
            return client.post(url, **kwargs)

        monkeypatch.setattr(httpx, method, fake)

    return _patch


#check_api_health


def test_health_ok_returns_body(patch_httpx) -> None:
    patch_httpx("get", 200, {"status": "ok", "model_version": "v2", "sklearn_version": "1.5.2"})
    ok, body, err = check_api_health("http://api:8000", 5.0)
    assert ok is True
    assert err is None
    assert body is not None and body["model_version"] == "v2"


def test_health_503_marks_unavailable(patch_httpx) -> None:
    patch_httpx("get", 503, {"detail": "modelo no cargado"})
    ok, body, err = check_api_health("http://api:8000", 5.0)
    assert ok is False
    assert err is not None and "503" in err


def test_health_connect_error_friendly_message(patch_httpx) -> None:
    patch_httpx("get", 0, raise_exc=httpx.ConnectError)
    ok, _body, err = check_api_health("http://api:8000", 5.0)
    assert ok is False
    assert err is not None and "conectar" in err.lower()


def test_health_timeout_friendly_message(patch_httpx) -> None:
    patch_httpx("get", 0, raise_exc=httpx.TimeoutException)
    ok, _body, err = check_api_health("http://api:8000", 5.0)
    assert ok is False
    assert err is not None and "imeout" in err


#call_predict


def test_predict_200_returns_body(patch_httpx) -> None:
    expected = {
        "churn": 1,
        "probability": 0.73,
        "model_version": "v2",
        "threshold": 0.441444,
        "request_id": "abc",
    }
    patch_httpx("post", 200, expected)
    status, body = call_predict("http://api:8000", {"x": 1}, 5.0)
    assert status == 200
    assert body == expected


def test_predict_422_propagates_status(patch_httpx) -> None:
    patch_httpx("post", 422, {"detail": "Invalid input parameters"})
    status, body = call_predict("http://api:8000", {"x": 1}, 5.0)
    assert status == 422
    assert "detail" in body


def test_predict_timeout_uses_zero_sentinel(patch_httpx) -> None:
    patch_httpx("post", 0, raise_exc=httpx.TimeoutException)
    status, body = call_predict("http://api:8000", {"x": 1}, 5.0)
    assert status == 0
    assert "imeout" in body["detail"]


def test_predict_connect_error_uses_zero_sentinel(patch_httpx) -> None:
    patch_httpx("post", 0, raise_exc=httpx.ConnectError)
    status, body = call_predict("http://api:8000", {"x": 1}, 5.0)
    assert status == 0


# --- build_payload_from_form ---


def test_payload_casts_numpy_to_native_python() -> None:
    #Streamlit puede devolver np.int64/np.float64; la API rechaza con strict=True
    np = pytest.importorskip("numpy")

    form_values = dict(DEFAULT_VALUES)
    form_values["tenure_months"] = np.int64(24)
    form_values["monthly_charge"] = np.float64(65.5)

    payload = build_payload_from_form(form_values)

    assert type(payload["tenure_months"]) is int
    assert type(payload["monthly_charge"]) is float
    assert type(payload["region"]) is str
    assert len(payload) == 15


def test_default_values_pass_api_contract() -> None:
    """Los defaults pre-cargados deben pasar la validacion strict de la API."""
    from app.schemas import CustomerInput

    validated = CustomerInput(**DEFAULT_VALUES)
    assert validated.region in ("centro", "norte", "oeste", "sur")
    assert validated.contract_type in ("anual", "bianual", "mensual")
