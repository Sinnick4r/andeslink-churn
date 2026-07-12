"""Contratos Pydantic de la API de inferencia

- ``CustomerInput``: contrato de entrada de ``/predict``. ``strict=True`` +
  ``extra="forbid"`` para prevenir mass-assignment, con rangos de negocio y ``Literal``
  en español.
  Los valores de las categóricas son los strings exactos del dataset con los que se entrenó el
  OneHotEncoder
- ``PredictionResponse``: contrato público de salida. Cambios incompatibles
  requieren versionar el endpoint (``/v1/predict`` → ``/v2/predict``)
- ``HealthResponse``: contrato del ``/health``
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Literals = valores exactos del dataset.
# Constituyen la Capa 1 contra categorias desconocidas:
# el cliente recibe 422 con feedback antes de llegar al modelo
ContractType = Literal["anual", "bianual", "mensual"]
PaymentMethod = Literal["credito", "debito", "efectivo", "transferencia"]
InternetService = Literal["cable", "fibra", "movil", "ninguno"]
Region = Literal["centro", "norte", "oeste", "sur"]


class CustomerInput(BaseModel):
    """Contrato de entrada de ``/predict``. Valida tipos + rangos de negocio.

    Nota sobre ``total_charges``: el modelo no lo usa directamente (se descarta
    en el ColumnTransformer con ``remainder="drop"``), pero
    ``add_derived_features`` lo necesita para calcular
    ``charges_per_month = total_charges / tenure_months``. Por eso es campo
    obligatorio del request
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    tenure_months: int = Field(ge=0, le=600, description="Antigüedad en meses")
    monthly_charge: float = Field(gt=0.0, le=10_000.0, description="Cargo mensual actual")
    total_charges: float = Field(
        ge=0.0, description="Cargos acumulados; requerido para la feature derivada"
    )
    support_tickets: int = Field(ge=0, le=100, description="Tickets de soporte abiertos")
    late_payments: int = Field(ge=0, le=120, description="Cantidad de pagos atrasados")
    avg_monthly_usage_gb: float = Field(ge=0.0, le=10_000.0, description="Uso mensual prom. (GB)")

    contract_type: ContractType
    payment_method: PaymentMethod
    internet_service: InternetService
    region: Region

    has_streaming: Literal[0, 1]
    has_security_pack: Literal[0, 1]
    num_products: int = Field(ge=1, le=10, description="Cantidad de productos contratados")
    customer_age: int = Field(ge=18, le=100, description="Edad del cliente")
    is_promo: Literal[0, 1]


class PredictionResponse(BaseModel):
    """Contrato público de salida de ``/predict``

    Devolver ``model_version`` y ``threshold`` permite que el cliente registre con
    qué condiciones se hizo la predicción - útil para post-mortem y comparación
    entre versiones del modelo.
    """

    # protected_namespaces=() habilita el campo ``model_version`` sin tirar el
    # warning de pydantic por el prefijo reservado ``model_``.
    model_config = ConfigDict(protected_namespaces=())

    churn: Literal[0, 1] = Field(description="Clase predicha")
    probability: float = Field(ge=0.0, le=1.0, description="Probabilidad de la clase positiva")
    model_version: str = Field(description="Versión del artefacto usado")
    threshold: float = Field(ge=0.0, le=1.0, description="Threshold aplicado para clasificar")
    request_id: str = Field(description="UUID4 del request, también presente en los logs")


class HealthResponse(BaseModel):
    # Contrato del ``/health``. ``status="degraded"`` ⇒ HTTP 503

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"] = Field(description="Estado operativo del servicio")
    model_loaded: bool = Field(description="True si el pipeline está cargado en memoria")
    model_version: str = Field(description="Versión del artefacto cargado")
    sklearn_version: str = Field(description="Versión de scikit-learn en runtime")
