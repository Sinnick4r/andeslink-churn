"""Carga y verificación de integridad del artefacto del modelo

- Verifica la versión de scikit-learn al startup: ante un mismatch, falla
  ruidosamente (la app no arranca) porque el ``.joblib`` es version-sensitive.
- Loguea el SHA-256 del artefacto para trazabilidad de integridad (nunca MD5/SHA-1).
- Solo se carga desde rutas configuradas (``MODEL_PATH``), nunca desde input de
  usuario (OWASP Deserialization: no deserializar fuentes no confiables).
"""

import hashlib
from pathlib import Path
from typing import Final

import joblib
import sklearn
from loguru import logger
from sklearn.pipeline import Pipeline

EXPECTED_SKLEARN_VERSION: Final[str] = "1.5.2"
_HASH_CHUNK_BYTES: Final[int] = 8192


def compute_model_hash(model_path: Path) -> str:
    """Calcula el SHA-256 del artefacto para verificación de integridad.

    Args:
        model_path: ruta al artefacto ``.joblib``.

    Returns:
        Digest hexadecimal de 64 caracteres.

    Raises:
        FileNotFoundError: si el artefacto no existe.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se puede hashear, no existe: {model_path}")

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            sha256.update(chunk)

    digest = sha256.hexdigest()
    assert len(digest) == 64, "Digest SHA-256 con longitud inesperada"
    return digest


def load_pipeline(model_path: Path) -> Pipeline:
    """Carga el pipeline serializado verificando versión de sklearn e integridad.

    Args:
        model_path: ruta al artefacto ``.joblib`` (preprocessor + classifier).

    Returns:
        El ``Pipeline`` de scikit-learn listo para inferencia.

    Raises:
        RuntimeError: si la versión de scikit-learn no coincide con la de entrenamiento.
        FileNotFoundError: si el artefacto no existe en la ruta configurada.
    """
    current_version = sklearn.__version__
    if current_version != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"sklearn version mismatch: esperado {EXPECTED_SKLEARN_VERSION}, "
            f"encontrado {current_version}. El artefacto puede ser incompatible."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Artefacto del modelo no encontrado en: {model_path}")

    model_hash = compute_model_hash(model_path)
    logger.info(f"Cargando modelo desde {model_path} | SHA256: {model_hash[:16]}...")

    pipeline = joblib.load(model_path)
    logger.info("Pipeline del modelo cargado correctamente")
    return pipeline
