# Configuración de logging con loguru



import sys

from loguru import logger

from app.config import settings


def configure_logging() -> None:
    #configura el sink global de loguru.
  
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
