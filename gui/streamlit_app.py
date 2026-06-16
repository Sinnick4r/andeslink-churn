"""GUI Streamlit que consume la API de inferencia de churn

Arquitectura:

- ``check_api_health`` y ``call_predict`` son funciones puras (sin dependencia
  de Streamlit) que encapsulan la comunicacion HTTP con la API. Esto las hace
  testeables con un ``httpx.MockTransport``.
- ``render_ui`` y ``render_result`` arman la interfaz y solo se invocan al
  ejecutar el modulo como app de Streamlit.
- ``API_URL`` se resuelve por variable de entorno, lo que permite apuntar a
  ``http://api:8000`` desde el container (red interna de compose) o a
  ``http://localhost:8000`` desde la maquina del desarrollador.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

# Resolucion de la URL de la API por entorno
# - En el compose: API_URL=http://api:8000 (red interna)
# - En local sin docker: cae al default localhost
API_URL: str = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS: float = 10.0

# Defaults pre-cargados: caso plausible del dataset
# Mismo payload que usan los tests de aceptacion de la API
DEFAULT_VALUES: dict[str, Any] = {
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

# Categorias permitidas: deben coincidir con los Literal del contrato de la API
# Si la API rechaza algo de esto, hay un mismatch que tenemos que arreglar
CONTRACT_TYPES: list[str] = ["anual", "bianual", "mensual"]
PAYMENT_METHODS: list[str] = ["credito", "debito", "efectivo", "transferencia"]
INTERNET_SERVICES: list[str] = ["cable", "fibra", "movil", "ninguno"]
REGIONS: list[str] = ["centro", "norte", "oeste", "sur"]
BINARY_LABELS: dict[int, str] = {0: "No", 1: "Si"}


def check_api_health(
    base_url: str, timeout: float
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Pre-flight check contra ``/health`` de la API.

    Args:
        base_url: URL base de la API, sin path
        timeout: segundos antes de abortar la conexion

    Returns:
        Tupla ``(ok, body, error_msg)``. Si la API responde 200, ``ok=True`` y
        ``body`` trae el JSON. Si falla, ``ok=False`` y ``error_msg`` describe
        el problema en lenguaje humano sin leak de detalles tecnicos
    """
    assert base_url, "base_url no puede ser vacio"
    assert timeout > 0, "timeout debe ser positivo"

    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
    except httpx.TimeoutException:
        return False, None, "Timeout consultando la API"
    except httpx.ConnectError:
        return False, None, "No se puede conectar con la API"
    except httpx.HTTPError as exc:
        return False, None, f"Error HTTP: {type(exc).__name__}"

    if response.status_code != 200:
        return False, None, f"la API respondio HTTP {response.status_code}"

    try:
        return True, response.json(), None
    except ValueError:
        return False, None, "Respuesta de la API no es JSON valido"


def call_predict(
    base_url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any]]:
    """Llama a ``POST /predict`` y devuelve ``(status_code, body)``.

    Usa ``status_code=0`` como sentinela cuando la request no llega a tener
    respuesta del servidor (timeout o problema de red). Asi el caller puede
    discriminar entre fallos de red y errores HTTP reales de la API.
    """
    assert base_url, "base_url no puede ser vacio"
    assert isinstance(payload, dict), "payload debe ser dict"

    try:
        response = httpx.post(f"{base_url}/predict", json=payload, timeout=timeout)
    except httpx.TimeoutException:
        return 0, {"detail": "Timeout al invocar /predict"}
    except httpx.ConnectError:
        return 0, {"detail": "No se puede conectar con la API"}
    except httpx.HTTPError as exc:
        return 0, {"detail": f"Error de red: {type(exc).__name__}"}

    try:
        body = response.json()
    except ValueError:
        body = {"detail": "Respuesta de la API no es JSON valido"}

    return response.status_code, body


def build_payload_from_form(form_values: dict[str, Any]) -> dict[str, Any]:
    """Normaliza los tipos del formulario al contrato Pydantic de la API.

    Streamlit puede devolver numericos como ``numpy.int64`` o ``float64``
    cuando vienen de ``number_input``. la API tiene ``strict=True`` y rechaza
    coerciones implicitas, asi que casteamos a int/float nativos de Python
    antes de serializar el JSON.
    """
    integer_fields = (
        "tenure_months",
        "support_tickets",
        "late_payments",
        "has_streaming",
        "has_security_pack",
        "num_products",
        "customer_age",
        "is_promo",
    )
    float_fields = ("monthly_charge", "total_charges", "avg_monthly_usage_gb")
    string_fields = ("contract_type", "payment_method", "internet_service", "region")

    payload: dict[str, Any] = {}
    for field in integer_fields:
        payload[field] = int(form_values[field])
    for field in float_fields:
        payload[field] = float(form_values[field])
    for field in string_fields:
        payload[field] = str(form_values[field])

    assert len(payload) == 15, f"Se esperaban 15 campos, se obtuvieron {len(payload)}"
    return payload


def render_result(status_code: int, body: dict[str, Any]) -> None:
    """Renderiza el resultado de la prediccion o el error apropiado."""
    if status_code == 200:
        churn = body["churn"]
        probability = body["probability"]
        threshold = body["threshold"]

        st.divider()
        if churn == 1:
            st.error("### Cliente en riesgo de churn")
        else:
            st.success("### Cliente con baja probabilidad de churn")

        col_prob, col_thr = st.columns([2, 1])
        col_prob.progress(
            min(max(probability, 0.0), 1.0),
            text=f"Probabilidad de abandono: {probability:.1%}",
        )
        col_thr.metric("Threshold de decision", f"{threshold:.4f}")

        with st.expander("Metadata de la prediccion"):
            st.code(
                f"churn_class: {churn}\n"
                f"probability: {probability:.6f}\n"
                f"threshold: {threshold}\n"
                f"model_version: {body['model_version']}\n"
                f"request_id: {body['request_id']}",
                language="text",
            )
        return

    if status_code == 422:
        st.warning(
            "la API rechazo el input (validacion). "
            "Revisar que todos los valores esten dentro de los rangos permitidos."
        )
        return

    if status_code == 0:
        st.error(f"Problema de red: {body.get('detail', 'sin detalle')}")
        return

    st.error(f"la API devolvio HTTP {status_code}: {body.get('detail', 'sin detalle')}")


def render_ui() -> None:
    """Renderiza la GUI completa: pre-flight, formulario y resultado."""
    st.set_page_config(page_title="AndesLink Churn", layout="centered")
    st.title("Predictor de Churn")
    st.caption("AndesLink Servicios Digitales S.A. - Inferencia individual de cliente")

    # Pre-flight: si la API no esta, no tiene sentido mostrar el formulario
    ok, health_body, err_msg = check_api_health(API_URL, REQUEST_TIMEOUT_SECONDS)
    if not ok or health_body is None:
        st.error(f"la API no esta disponible: {err_msg}")
        st.info(f"Revisar que el servicio este corriendo en `{API_URL}`")
        st.stop()

    # Badge de status arriba con info del modelo cargado
    col_a, col_b, col_c = st.columns(3)
    col_a.success("API conectada")
    col_b.metric("Modelo", health_body["model_version"])
    col_c.metric("scikit-learn", health_body["sklearn_version"])

    st.divider()

    # Formulario con submit batch: el rerun de Streamlit solo dispara al apretar el boton
    with st.form("predict_form"):
        st.subheader("Datos del cliente")
        col1, col2 = st.columns(2)
        tenure_months = col1.number_input(
            "Antiguedad (meses)",
            min_value=0,
            max_value=600,
            value=DEFAULT_VALUES["tenure_months"],
            step=1,
        )
        customer_age = col2.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=DEFAULT_VALUES["customer_age"],
            step=1,
        )
        region = col1.selectbox(
            "Region",
            REGIONS,
            index=REGIONS.index(DEFAULT_VALUES["region"]),
        )
        is_promo = col2.selectbox(
            "En promocion?",
            options=[0, 1],
            format_func=lambda x: BINARY_LABELS[x],
            index=DEFAULT_VALUES["is_promo"],
        )

        st.subheader("Servicio contratado")
        col3, col4 = st.columns(2)
        contract_type = col3.selectbox(
            "Tipo de contrato",
            CONTRACT_TYPES,
            index=CONTRACT_TYPES.index(DEFAULT_VALUES["contract_type"]),
        )
        payment_method = col4.selectbox(
            "Metodo de pago",
            PAYMENT_METHODS,
            index=PAYMENT_METHODS.index(DEFAULT_VALUES["payment_method"]),
        )
        internet_service = col3.selectbox(
            "Servicio de internet",
            INTERNET_SERVICES,
            index=INTERNET_SERVICES.index(DEFAULT_VALUES["internet_service"]),
        )
        num_products = col4.number_input(
            "Productos contratados",
            min_value=1,
            max_value=10,
            value=DEFAULT_VALUES["num_products"],
            step=1,
        )
        has_streaming = col3.selectbox(
            "Tiene streaming?",
            options=[0, 1],
            format_func=lambda x: BINARY_LABELS[x],
            index=DEFAULT_VALUES["has_streaming"],
        )
        has_security_pack = col4.selectbox(
            "Pack de seguridad?",
            options=[0, 1],
            format_func=lambda x: BINARY_LABELS[x],
            index=DEFAULT_VALUES["has_security_pack"],
        )

        st.subheader("Comportamiento y consumo")
        col5, col6 = st.columns(2)
        monthly_charge = col5.number_input(
            "Cargo mensual ($)",
            min_value=0.01,
            max_value=10_000.0,
            value=DEFAULT_VALUES["monthly_charge"],
            step=1.0,
            format="%.2f",
        )
        total_charges = col6.number_input(
            "Cargos acumulados ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=DEFAULT_VALUES["total_charges"],
            step=10.0,
            format="%.2f",
        )
        support_tickets = col5.number_input(
            "Tickets de soporte",
            min_value=0,
            max_value=100,
            value=DEFAULT_VALUES["support_tickets"],
            step=1,
        )
        late_payments = col6.number_input(
            "Pagos atrasados",
            min_value=0,
            max_value=120,
            value=DEFAULT_VALUES["late_payments"],
            step=1,
        )
        avg_monthly_usage_gb = col5.number_input(
            "Uso mensual prom. (GB)",
            min_value=0.0,
            max_value=10_000.0,
            value=DEFAULT_VALUES["avg_monthly_usage_gb"],
            step=1.0,
            format="%.2f",
        )

        submitted = st.form_submit_button(
            "Predecir churn", type="primary", use_container_width=True
        )

    if not submitted:
        return

    form_values = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "region": region,
        "has_streaming": has_streaming,
        "has_security_pack": has_security_pack,
        "num_products": num_products,
        "customer_age": customer_age,
        "is_promo": is_promo,
    }
    payload = build_payload_from_form(form_values)

    with st.spinner("Consultando modelo..."):
        status_code, body = call_predict(API_URL, payload, REQUEST_TIMEOUT_SECONDS)

    render_result(status_code, body)


if __name__ == "__main__":
    render_ui()
