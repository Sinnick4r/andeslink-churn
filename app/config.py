"""config de la API: no hay credenciales, rutas absolutas ni cosas
hardcodeadas. Unica instancia inmutable ``settings`` al
importar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Defaults del proyecto.
# El threshold canónico vive en ``reports/train_metrics.json`` (``selected_threshold``),
# pero ese archivo se excluye del container de runtime (``.dockerignore``),
# por eso el valor calibrado se inyecta por entorno (compose) con este default.
DEFAULT_MODEL_PATH: Final[str] = "models/churn_model_v2.joblib"
DEFAULT_MODEL_VERSION: Final[str] = "v2"
DEFAULT_THRESHOLD: Final[float] = 0.441444  # F1-óptimo en val (LogisticRegression, E1)
DEFAULT_ENV: Final[str] = "dev"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# Registro de inferencias para el monitoreo de datos (Evidently)
# Persiste las features de entrada de cada prediccion en JSONL, en un almacen
# separado del log operacional. Se apaga con INFERENCE_LOG_ENABLED=false
DEFAULT_INFERENCE_LOG_ENABLED: Final[str] = "true"
DEFAULT_INFERENCE_LOG_PATH: Final[str] = "monitoring/data/inferences.jsonl"

# valores que se interpretan como verdadero al leer el flag desde el entorno
_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuración inmutable de la aplicación, resuelta desde el entorno."""

    model_path: Path
    model_version: str
    threshold: float
    env: str
    log_level: str
    inference_log_enabled: bool
    inference_log_path: Path

    def __post_init__(self) -> None:
        # Validaciones de invariante ≥2 assertions por función no trivial).
        assert 0.0 <= self.threshold <= 1.0, f"threshold fuera de [0,1]: {self.threshold}"
        assert self.model_version, "model_version no puede ser vacío"

    @property
    def is_production(self) -> bool:
        return self.env == "production"


def load_settings() -> Settings:
    """Resuelve la configuración desde variables de entorno.

    Raises:
        ValueError: si ``THRESHOLD`` no es convertible a float.
    """
    return Settings(
        model_path=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)),
        model_version=os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        threshold=float(os.environ.get("THRESHOLD", DEFAULT_THRESHOLD)),
        env=os.environ.get("ENV", DEFAULT_ENV),
        log_level=os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        inference_log_enabled=os.environ.get("INFERENCE_LOG_ENABLED", DEFAULT_INFERENCE_LOG_ENABLED)
        .strip()
        .lower()
        in _TRUE_VALUES,
        inference_log_path=Path(os.environ.get("INFERENCE_LOG_PATH", DEFAULT_INFERENCE_LOG_PATH)),
    )


settings: Final[Settings] = load_settings()
