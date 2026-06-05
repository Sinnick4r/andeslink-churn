"""API de inferencia de churn — FastAPI.

Wiring:
- ``lifespan`` carga el modelo una sola vez al startup; si falla, la app no arranca.
- ``/predict`` (solo POST): valida con Pydantic, aplica feature engineering y devuelve
  el contrato ``PredictionResponse``.
- ``/health`` (solo GET): 200 si el modelo está cargado, 503 si no.
- Middleware de ``request_id`` (UUID4) propagado a logs y header ``X-Request-ID``.
- Exception handlers que devuelven mensajes genéricos al cliente (el detalle va al log).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.logging_config import configure_logging
from app.model_loader import load_pipeline
from app.schemas import CustomerInput, HealthResponse, PredictionResponse

# add_derived_features se importa del MISMO módulo usado en training (src/features.py).
# Single source of truth -> sin training/serving skew: la feature engineering en
# serving es idéntica byte a byte a la de entrenamiento.
# Nota de despliegue: el Dockerfile del API debe copiar src/features.py al container.
from src.features import add_derived_features


@asynccontextmanager
async def lifespan(app: FastAPI):
    # carga el modelo al startup y deja la referencia en ``app.state``
    configure_logging()
    logger.info("Startup: cargando artefacto del modelo")
    # Si load_pipeline hace raise (versión incompatible o archivo ausente),
    # el startup se corta y la app NO arranca — comportamiento deseado.
    app.state.pipeline = load_pipeline(settings.model_path)
    app.state.threshold = settings.threshold
    app.state.model_version = settings.model_version
    logger.info(f"Modelo {settings.model_version} listo | threshold={settings.threshold}")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="AndesLink Churn Prediction API",
    version=settings.model_version,
    lifespan=lifespan,
    # /docs, /redoc y /openapi.json deshabilitados en produccion
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Asigna un request_id (UUID4) y lo propaga a logs y al header de respuesta
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    with logger.contextualize(request_id=request_id):
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 422 genérico al cliente; el detalle de validación queda en el log
    logger.warning(f"Input inválido en {request.url.path}: {len(exc.errors())} error(es)")
    return JSONResponse(status_code=422, content={"detail": "Invalid input parameters"})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # 500 genérico: nunca se expone stack trace, ruta interna ni mensaje raw
    logger.error(f"Excepción no manejada en {request.url.path}: {type(exc).__name__}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def run_inference(pipeline: Any, customer: CustomerInput) -> tuple[int, float]:
    """aplica feature engineering + pipeline y devuelve ``(clase, probabilidad)``.

    ``add_derived_features`` se aplica aca porque el ``.joblib`` serializado no la
    incluye: el pipeline es ``preprocessor + classifier`` y espera
    ``charges_per_month`` / ``tickets_per_year`` ya calculadas en el DataFrame.

    Args:
        pipeline: pipeline de sklearn cargado (con ``predict_proba``).
        customer: input validado por Pydantic.

    Returns:
        ``(churn, probability)`` — clase binaria según el threshold y probabilidad
        de la clase positiva.
    """
    frame = pd.DataFrame([customer.model_dump()])
    frame = add_derived_features(frame)
    probability = float(pipeline.predict_proba(frame)[0][1])

    # Pre/postcondiciones

    assert frame.shape[0] == 1
    assert 0.0 <= probability <= 1.0
    churn = int(probability >= settings.threshold)
    return churn, probability


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    # predice churn para un cliente. Solo acepta POST (GET ⇒ 405 automático)
    start = time.perf_counter()
    churn, probability = run_inference(request.app.state.pipeline, customer)
    elapsed = time.perf_counter() - start
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Solo metadatos de operación; nunca features del cliente
    logger.bind(
        churn_class=churn,
        probability=round(probability, 4),
        duration_ms=round(elapsed * 1000, 2),
        model_version=settings.model_version,
    ).info("Prediction completed")

    return PredictionResponse(
        churn=churn,
        probability=probability,
        model_version=request.app.state.model_version,
        threshold=request.app.state.threshold,
        request_id=request_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> JSONResponse:
    # Readiness real: 503 si el modelo no esta cargado
    pipeline = getattr(request.app.state, "pipeline", None)
    is_loaded = pipeline is not None
    body = HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=getattr(request.app.state, "model_version", "unknown"),
        sklearn_version=sklearn.__version__,
    )
    return JSONResponse(content=body.model_dump(), status_code=200 if is_loaded else 503)
