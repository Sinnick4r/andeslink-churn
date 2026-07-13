# test para valdiar monitoreo ya levantado con Docker Compose

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

API_URL = os.getenv("SMOKE_API_URL", "http://localhost:8000")
GUI_URL = os.getenv("SMOKE_GUI_URL", "http://localhost:8501")
PROMETHEUS_URL = os.getenv("SMOKE_PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("SMOKE_GRAFANA_URL", "http://localhost:3000")
INFERENCE_LOG = Path("monitoring/data/inferences.jsonl")
TIMEOUT_SECONDS = 90

PAYLOAD = {
    "tenure_months": 24,
    "monthly_charge": 65.5,
    "total_charges": 1572.0,
    "support_tickets": 2,
    "late_payments": 1,
    "avg_monthly_usage_gb": 120.0,
    "contract_type": "anual",
    "payment_method": "credito",
    "internet_service": "fibra",
    "region": "centro",
    "has_streaming": 1,
    "has_security_pack": 0,
    "num_products": 2,
    "customer_age": 35,
    "is_promo": 0,
}


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _wait_for(name: str, check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            if check():
                print(f"[ok] {name}")
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"timeout esperando {name}{detail}")


def _prometheus_target_is_up() -> bool:
    query = urllib.parse.urlencode({"query": 'up{job="churn-api"}'})
    data = _request_json(f"{PROMETHEUS_URL}/api/v1/query?{query}")
    results = data.get("data", {}).get("result", [])
    return bool(results and float(results[0]["value"][1]) == 1.0)


def _log_contains(request_id: str) -> bool:
    if not INFERENCE_LOG.exists():
        return False
    return request_id in INFERENCE_LOG.read_text(encoding="utf-8")


def main() -> int:
    _wait_for("API", lambda: _request_json(f"{API_URL}/health").get("status") == "ok")
    _wait_for("GUI", lambda: _request_text(f"{GUI_URL}/_stcore/health").strip() == "ok")
    _wait_for("Prometheus", lambda: _request_text(f"{PROMETHEUS_URL}/-/ready").strip() != "")
    _wait_for("Grafana", lambda: _request_json(f"{GRAFANA_URL}/api/health").get("database") == "ok")

    prediction = _request_json(f"{API_URL}/predict", PAYLOAD)
    request_id = str(prediction["request_id"])
    assert prediction["churn"] in (0, 1)
    assert 0.0 <= float(prediction["probability"]) <= 1.0
    print("[ok] prediccion")

    metrics = _request_text(f"{API_URL}/metrics")
    assert "churn_api_predictions_total" in metrics
    assert "churn_api_inference_duration_seconds" in metrics
    print("[ok] metricas custom")

    _wait_for("target churn-api en Prometheus", _prometheus_target_is_up)
    _wait_for("registro de inferencia", lambda: _log_contains(request_id))

    print("Smoke test de monitoreo completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
