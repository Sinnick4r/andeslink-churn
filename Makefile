# Makefile del Proyecto AndesLink Churn MLOps
# Comandos para desarrollo local y CI.
# Uso: `make <target>` o `make help` para listar opciones.

.PHONY: help check fix format smoke test all clean pull up-build up

# Target default: muestra ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  make check    - Linting con ruff (no modifica archivos, falla si hay issues)"
	@echo "  make fix      - Linting con ruff + autofix (modifica archivos)"
	@echo "  make format   - Formateo con ruff format (modifica archivos)"
	@echo "  make smoke    - Smoke tests de modulos src/"
	@echo "  make test     - Suite completa de pytest (cuando exista tests/)"
	@echo "  make all      - format + check + smoke (orden de aplicacion seguro)"
	@echo "  make clean    - Eliminar archivos generados (cache, .pyc, etc)"

# Linting estricto: no modifica nada, exit 1 si hay issues
check:
	ruff check .

# Linting con autofix: arregla lo que puede automaticamente
fix:
	ruff check . --fix

# Formateo de codigo
format:
	ruff format .

# Smoke tests de modulos src/ (no requiere CSV ni modelo)
smoke:
	python src/features.py

# Suite de pytest (solo corre si existe carpeta tests/)
test:
	pytest
# Pipeline completo en orden seguro: format primero, despues check
all: format fix check smoke

# Limpieza de archivos generados
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
pull:
	dvc pull models/churn_model_v2.joblib

up-build: 
	docker compose up -d --build

up: 
	docker compose up -d