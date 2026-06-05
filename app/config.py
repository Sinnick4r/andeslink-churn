"""Configuración central de la API.

Toda la configuración git log: no hay credenciales, rutas absolutas ni secretos
hardcodeados. Se materializa una única instancia inmutable ``settings`` al
importar el módulo.
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


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuración inmutable de la aplicación, resuelta desde el entorno."""

    model_path: Path
    model_version: str
    threshold: float
    env: str
    log_level: str

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
    )


settings: Final[Settings] = load_settings()
