"""
Registro de inferencias para el monitoreo de datos
Graba en un JSON las features de entrada y el resultado de cada prediccion,
para analizar el data drift con evidently. El request_id es el puente de
trazabilidad entre los dos planos el json y el log
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

# serializa los appends para evitar lineas entrelazadas ante requests concurrentes
_write_lock = Lock()


def record_inference(
    path: Path,
    features: dict[str, Any],
    churn: int,
    probability: float,
    threshold: float,
    model_version: str,
    request_id: str,
) -> bool:
    # Escribe una linea JSONL con la inferencia, retorna True si persistio

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "threshold": threshold,
        "churn": churn,
        "probability": round(probability, 6),
        **features,
    }
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        # best-effort: no patea la prediccion por un fallo de registro
        logger.warning(f"No se pudo registrar la inferencia (request_id={request_id}): {exc}")
        return False
