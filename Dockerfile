# syntax=docker/dockerfile:1.7

# Stage 1: builder
# Instala las dependencias del runtime en un venv aislado que despues se
# transplanta al stage final, asi la imagen runtime no carga ni pip ni cache

FROM python:3.11-slim-bookworm AS builder

# Variables para que pip no genere cache ni warnings en el container
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# venv en una ruta independiente del usuario
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip pinneado por reproducibilidad del build
RUN pip install pip==24.2

# Solo requirements de runtime (sin httpx, sin tooling de dev)
COPY requirements.txt .
RUN pip install -r requirements.txt


# Stage 2: runtime
# Imagen final. Solo Python, el venv ya armado, el codigo de la API
# y el artefacto del modelo. Corre como usuario no-root sin shell

FROM python:3.11-slim-bookworm AS runtime

# Aplicar parches de seguridad del SO sobre la base slim
RUN apt-get update \
    && apt-get -y --no-install-recommends upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Usuario y grupo del sistema sin home y sin shell de login
RUN groupadd --system appgroup \
    && useradd --system \
        --gid appgroup \
        --no-create-home \
        --shell /usr/sbin/nologin \
        appuser

# Copia del venv ya construido en el builder
COPY --from=builder /opt/venv /opt/venv

# Variables de runtime
# PYTHONDONTWRITEBYTECODE evita escribir .pyc (necesario con filesystem read-only)
# PYTHONUNBUFFERED garantiza que los logs salgan inmediatamente a stdout
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


# Solo el codigo y el artefacto que la API necesita en runtime
# src/ es necesario porque main.py importa add_derived_features desde src.features
# para compartir la misma feature engineering que se usa en training
COPY app/ ./app/
COPY src/ ./src/
COPY models/churn_model_v2.joblib ./models/churn_model_v2.joblib

# appuser solo necesita lectura sobre el codigo y el modelo
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck por si el container corre fuera de compose
# El compose define el suyo y lo sobrescribe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)" \
    || exit 1

# uvicorn sin access log: la app ya loguea /predict con request_id por loguru,
# y /health se llama cada 30s por el healthcheck (no inflar logs con eso)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
