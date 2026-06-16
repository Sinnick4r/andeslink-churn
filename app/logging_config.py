"""Configuración de logging con loguru (guía 8.5).

- JSON estructurado (``serialize=True``) en producción; human-readable en dev.
- Sink a ``stdout`` para que ``docker logs`` capture la salida.
- Nivel mínimo configurable por ``LOG_LEVEL``.
- Nunca usar ``print()`` en código de producción.

Política de contenido (guía 3.4 / OWASP Logging / NIST AU-3): se loguean
metadatos de operación (request_id, clase, probabilidad, duración, versión del
modelo) pero NUNCA valores de features del cliente (posible PII) ni secretos.
"""

import sys

from loguru import logger

from app.config import settings


def configure_logging() -> None:
    """Configura el sink global de loguru.

    Side effect (guía 2.4): reemplaza todos los handlers previos de loguru.
    Idempotente--> puede llamarse más de una vez sin acumular sinks
    """
    logger.remove()

    if settings.is_production:
        logger.add(sys.stdout, level=settings.log_level, serialize=True)
        return

    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
    )
