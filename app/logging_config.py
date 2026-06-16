"""Configuración de logging con loguru .

- JSON estructurado (``serialize=True``) en producción; human-readable en dev.
- Sink a ``stdout`` para que ``docker logs`` capture la salida.
- Nivel mínimo configurable por ``LOG_LEVEL``.
- Nunca usar ``print()`` en código de producción.
"""

import sys

from loguru import logger

from app.config import settings


def configure_logging() -> None:
    """Configura el sink global de loguru.

    Side effect: reemplaza todos los handlers previos de loguru.
    Idempotente--> puede llamarse varias veces sin acumularse
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
