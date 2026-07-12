"""Generador de trafico para poblar el monitoreo de AndesLink Churn

Manda requests a /predict para alimentar al mismo tiempo las metricas de
Prometheus/Grafana (requests, status, latencia, predicciones por clase) y el
registro de inferencias que consume Evidently para el reporte de drift.

Modos:
- batch: una tanda sana y una con drift, para generar el registro del reporte
- loop: trafico sano de fondo a tasa fija, para monitorear en tiempo real
- escenarios: tramos controlados para demostrar paneles y alertas

Escenarios disponibles:
- baseline: trafico sano estable
- burst: pico de carga
- churn-spike: perfil de alto riesgo, dispara la proporcion de churn predicho
- drift: corrimiento configurable de las features de entrada
- errors: payloads invalidos para disparar 422 y mover la tasa de error
- demo: secuencia baseline, burst, churn-spike, drift, errors, recovery

churn-spike y drift usan el mismo mecanismo de corrimiento a distinta dosis:
drift con severidad media, churn-spike con severidad alta. No son dos fenomenos
distintos, son el mismo corrimiento de entrada visto en dos intensidades.

Las features se castean a tipos nativos de Python porque el contrato de la API
usa strict=True y rechaza numpy.int64 / numpy.float64.

Uso:
    python monitoring/generate_traffic.py --healthy 300 --drift 200
    python monitoring/generate_traffic.py --loop --rate 2
    python monitoring/generate_traffic.py --scenario demo --demo-scale 0.75
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd

# columnas de entrada por tipo, para castear desde el DataFrame al payload JSON
_INT_FEATURES = [
    "tenure_months",
    "support_tickets",
    "late_payments",
    "has_streaming",
    "has_security_pack",
    "num_products",
    "customer_age",
    "is_promo",
]
_FLOAT_FEATURES = ["monthly_charge", "total_charges", "avg_monthly_usage_gb"]
_STR_FEATURES = ["contract_type", "payment_method", "internet_service", "region"]
_ALL_FEATURES = _INT_FEATURES + _FLOAT_FEATURES + _STR_FEATURES

DEFAULT_CSV = Path("data/raw/churn_sintetico.csv")
DEFAULT_URL = "http://localhost:8000"

# cada cuantos segundos cambia la tasa cuando el tramo usa un rango min/max
RATE_CHANGE_SECONDS = 10.0

PayloadKind = Literal["healthy", "drift", "churn_spike", "invalid"]
DriftSeverity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class SendResult:
    """Resultado compacto de un request individual"""

    status_code: int | None
    ok: bool
    expected_error: bool = False
    churn: int | None = None
    probability: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class Phase:
    """Tramo de trafico dentro del escenario demo"""

    name: str
    duration_seconds: float
    kind: PayloadKind
    rate: float | None = None
    min_rate: float | None = None
    max_rate: float | None = None
    expected_statuses: tuple[int, ...] = (200,)


# parametros de corrimiento por severidad, compartidos por drift y churn-spike
_SHIFT_PROFILES: dict[DriftSeverity, dict[str, Any]] = {
    "low": {
        "tenure_factor": (0.75, 0.90),
        "tickets_delta": (0, 2),
        "late_delta": (0, 1),
        "charge_factor": (1.03, 1.10),
        "monthly_share": 0.25,
        "security_zero_share": 0.15,
    },
    "medium": {
        "tenure_factor": (0.55, 0.80),
        "tickets_delta": (1, 4),
        "late_delta": (1, 3),
        "charge_factor": (1.10, 1.25),
        "monthly_share": 0.60,
        "security_zero_share": 0.30,
    },
    "high": {
        "tenure_factor": (0.20, 0.45),
        "tickets_delta": (5, 10),
        "late_delta": (4, 8),
        "charge_factor": (1.35, 1.70),
        "monthly_share": 0.90,
        "security_zero_share": 0.70,
    },
}


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    # castea cada feature al tipo nativo que exige el contrato strict de la API
    payload: dict[str, Any] = {}
    for col in _INT_FEATURES:
        payload[col] = int(row[col])
    for col in _FLOAT_FEATURES:
        payload[col] = float(row[col])
    for col in _STR_FEATURES:
        payload[col] = str(row[col])
    return payload


def _sample_payload(df: pd.DataFrame, rng: random.Random) -> dict[str, Any]:
    # muestrea una fila al azar con reemplazo y la castea al contrato
    idx = rng.randrange(len(df))
    return _row_to_payload(df.iloc[idx])


def _apply_shift(
    payload: dict[str, Any],
    rng: random.Random,
    severity: DriftSeverity,
) -> dict[str, Any]:
    """Corre las features de entrada en la direccion de mayor riesgo

    Mecanismo unico para drift y churn-spike: la severidad regula la dosis. Mueve
    tenure, tickets, pagos atrasados, cargo y la mezcla categorica respetando los
    rangos del contrato. No construye un cliente desde cero, parte de una fila real.
    """
    profile = _SHIFT_PROFILES[severity]
    shifted = dict(payload)

    tenure_factor = rng.uniform(*profile["tenure_factor"])
    charge_factor = rng.uniform(*profile["charge_factor"])
    tickets_delta = rng.randint(*profile["tickets_delta"])
    late_delta = rng.randint(*profile["late_delta"])

    shifted["tenure_months"] = max(0, int(int(payload["tenure_months"]) * tenure_factor))
    shifted["support_tickets"] = min(100, int(payload["support_tickets"]) + tickets_delta)
    shifted["late_payments"] = min(120, int(payload["late_payments"]) + late_delta)
    shifted["monthly_charge"] = min(
        10_000.0, round(float(payload["monthly_charge"]) * charge_factor, 2)
    )

    # mantiene coherencia aproximada entre cargo mensual y total acumulado
    tenure = max(int(shifted["tenure_months"]), 1)
    if rng.random() < 0.70:
        total = float(shifted["monthly_charge"]) * tenure * rng.uniform(0.75, 1.15)
        shifted["total_charges"] = round(total, 2)

    # corrimiento categorico: cambia la mezcla sin forzar todos los casos
    if rng.random() < profile["monthly_share"]:
        shifted["contract_type"] = "mensual"
    if rng.random() < profile["security_zero_share"]:
        shifted["has_security_pack"] = 0
    if rng.random() < 0.25:
        shifted["payment_method"] = rng.choice(["efectivo", "transferencia", "debito"])

    return shifted


def _make_invalid_payload(payload: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Genera payloads invalidos para disparar 422 y mover la tasa de error

    El contrato Pydantic usa strict=True, rangos y categorias Literal, asi que
    estos casos deberian ser rechazados antes de llegar al modelo.
    """
    bad = dict(payload)
    choice = rng.choice(["bad_type", "bad_range", "bad_category", "extra_field"])

    if choice == "bad_type":
        bad["tenure_months"] = "doce"  # strict rechaza str donde espera int
    elif choice == "bad_range":
        bad["monthly_charge"] = -10.0  # viola gt=0.0
    elif choice == "bad_category":
        bad["contract_type"] = "semanal"  # fuera del Literal
    else:
        bad["customer_id"] = "CUST-ERROR-001"  # viola extra=forbid

    return bad


def _build_payload(
    df: pd.DataFrame,
    rng: random.Random,
    kind: PayloadKind,
    drift_severity: DriftSeverity = "medium",
) -> dict[str, Any]:
    # construye el payload segun el tipo de trafico del tramo
    payload = _sample_payload(df, rng)

    if kind == "healthy":
        return payload
    if kind == "drift":
        return _apply_shift(payload, rng, drift_severity)
    if kind == "churn_spike":
        return _apply_shift(payload, rng, "high")
    if kind == "invalid":
        return _make_invalid_payload(payload, rng)

    raise ValueError(f"tipo de payload no soportado: {kind}")


def _send(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    expected_statuses: tuple[int, ...] = (200,),
) -> SendResult:
    # envia un request a /predict y resume el resultado
    try:
        resp = client.post(f"{url}/predict", json=payload, timeout=10.0)
    except httpx.HTTPError as exc:
        return SendResult(status_code=None, ok=False, error=str(exc))

    ok = resp.status_code == 200
    expected_error = resp.status_code in expected_statuses and not ok

    churn: int | None = None
    probability: float | None = None
    if ok:
        try:
            body = resp.json()
            churn = int(body["churn"])
            probability = float(body["probability"])
        except (KeyError, TypeError, ValueError):
            # la metrica ya fue emitida; el parseo es solo para consola
            pass

    return SendResult(
        status_code=resp.status_code,
        ok=ok,
        expected_error=expected_error,
        churn=churn,
        probability=probability,
    )


def _check_health(client: httpx.Client, url: str) -> bool:
    # verificacion previa de disponibilidad de la API
    try:
        health = client.get(f"{url}/health", timeout=5.0)
    except httpx.HTTPError as exc:
        print(f"No se pudo conectar a la API en {url}: {exc}", file=sys.stderr)
        return False
    if health.status_code != 200:
        print(f"La API respondio /health con {health.status_code}", file=sys.stderr)
        return False
    return True


def _validate_rate(rate: float | None, min_rate: float | None, max_rate: float | None) -> None:
    # valida que las tasas sean positivas y el rango coherente
    for name, value in (("--rate", rate), ("--min-rate", min_rate), ("--max-rate", max_rate)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} debe ser mayor que 0")
    if min_rate is not None and max_rate is not None and min_rate > max_rate:
        raise ValueError("--min-rate no puede ser mayor que --max-rate")


def _next_rate(
    rng: random.Random,
    fixed_rate: float | None,
    min_rate: float | None,
    max_rate: float | None,
) -> float:
    # tasa fija, o aleatoria dentro del rango si se paso min/max
    if min_rate is not None and max_rate is not None:
        return rng.uniform(min_rate, max_rate)
    return fixed_rate if fixed_rate is not None else 2.0


def _print_summary(
    label: str,
    sent: int,
    ok: int,
    expected_errors: int,
    churn_1: int,
    probabilities: list[float],
    started_at: float,
) -> None:
    # imprime el resumen de un tramo con tasa real y proporcion de churn
    elapsed = max(time.perf_counter() - started_at, 0.001)
    fail = sent - ok - expected_errors
    avg_prob = sum(probabilities) / len(probabilities) if probabilities else None
    avg_prob_txt = f"{avg_prob:.3f}" if avg_prob is not None else "n/a"
    churn_rate = churn_1 / ok if ok else 0.0
    print(
        f"[{label}] enviados={sent} | ok={ok} | "
        f"422 esperados={expected_errors} | fallidos={fail} | "
        f"churn_1={churn_1} ({churn_rate:.1%}) | "
        f"prob_prom={avg_prob_txt} | tasa_real={sent / elapsed:.2f} req/s"
    )


def _run_phase(
    client: httpx.Client,
    url: str,
    df: pd.DataFrame,
    rng: random.Random,
    *,
    label: str,
    kind: PayloadKind,
    rate: float | None = None,
    min_rate: float | None = None,
    max_rate: float | None = None,
    max_duration: float | None = None,
    expected_statuses: tuple[int, ...] = (200,),
    drift_severity: DriftSeverity = "medium",
) -> int:
    """Corre un tramo de trafico de un tipo dado

    Con ``max_duration`` corta al cumplir esos segundos; con ``None`` corre hasta
    Ctrl+C. Si se pasan min/max la tasa varia cada RATE_CHANGE_SECONDS, si no usa
    la tasa fija. Unica funcion para el loop de fondo y para los escenarios.
    """
    _validate_rate(rate, min_rate, max_rate)

    sent = ok = expected_errors = churn_1 = 0
    probabilities: list[float] = []

    started_at = time.perf_counter()
    last_report = last_rate_change = started_at
    current_rate = _next_rate(rng, rate, min_rate, max_rate)
    variable = min_rate is not None and max_rate is not None

    print(f"\n[{label}] inicio | tipo={kind} | tasa={current_rate:.2f} req/s")
    try:
        while max_duration is None or time.perf_counter() - started_at < max_duration:
            request_started = time.perf_counter()

            if variable and request_started - last_rate_change >= RATE_CHANGE_SECONDS:
                current_rate = _next_rate(rng, rate, min_rate, max_rate)
                last_rate_change = request_started
                print(f"[{label}] nueva tasa objetivo: {current_rate:.2f} req/s")

            payload = _build_payload(df, rng, kind, drift_severity)
            result = _send(client, url, payload, expected_statuses=expected_statuses)

            sent += 1
            ok += int(result.ok)
            expected_errors += int(result.expected_error)
            churn_1 += int(result.churn == 1)
            if result.probability is not None:
                probabilities.append(result.probability)

            if time.perf_counter() - last_report >= 10.0:
                _print_summary(label, sent, ok, expected_errors, churn_1, probabilities, started_at)
                last_report = time.perf_counter()

            sleep_for = 1.0 / current_rate - (time.perf_counter() - request_started)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario")

    _print_summary(label, sent, ok, expected_errors, churn_1, probabilities, started_at)
    return 0 if sent == ok + expected_errors else 2


def _run_batch(
    client: httpx.Client,
    url: str,
    df: pd.DataFrame,
    rng: random.Random,
    healthy: int,
    drift: int,
    drift_severity: DriftSeverity = "medium",
) -> int:
    # modo batch compatible: una tanda sana y una con drift para el reporte
    if healthy < 0 or drift < 0:
        print("--healthy y --drift no pueden ser negativos", file=sys.stderr)
        return 1

    ok = total = churn_1 = 0
    probabilities: list[float] = []

    for label, kind, count in (("sanos", "healthy", healthy), ("con drift", "drift", drift)):
        print(f"Enviando {count} requests {label}...")
        for _ in range(count):
            result = _send(client, url, _build_payload(df, rng, kind, drift_severity))
            total += 1
            ok += int(result.ok)
            churn_1 += int(result.churn == 1)
            if result.probability is not None:
                probabilities.append(result.probability)

    avg_prob = sum(probabilities) / len(probabilities) if probabilities else None
    avg_prob_txt = f"{avg_prob:.3f}" if avg_prob is not None else "n/a"
    churn_rate = churn_1 / ok if ok else 0.0
    print(
        f"Listo: {ok}/{total} requests OK | "
        f"churn_1={churn_1} ({churn_rate:.1%}) | prob_prom={avg_prob_txt}"
    )
    return 0 if ok == total else 2


def _demo_phases(scale: float) -> list[Phase]:
    # secuencia de tramos para la demo, con un piso de 10s por tramo
    if scale <= 0:
        raise ValueError("--demo-scale debe ser mayor que 0")

    def secs(value: float) -> float:
        return max(10.0, value * scale)

    return [
        Phase("baseline", secs(45), "healthy", rate=2.0),
        Phase("burst", secs(35), "healthy", min_rate=8.0, max_rate=12.0),
        Phase("churn-spike", secs(60), "churn_spike", rate=4.0),
        Phase("drift", secs(60), "drift", rate=3.0),
        Phase("errors-422", secs(35), "invalid", rate=1.5, expected_statuses=(422,)),
        Phase("recovery", secs(45), "healthy", rate=2.0),
    ]


def _run_demo(
    client: httpx.Client,
    url: str,
    df: pd.DataFrame,
    rng: random.Random,
    args: argparse.Namespace,
) -> int:
    # encadena los tramos de la demo con una pausa opcional entre ellos
    exit_code = 0
    phases = _demo_phases(args.demo_scale)
    for phase in phases:
        code = _run_phase(
            client,
            url,
            df,
            rng,
            label=phase.name,
            kind=phase.kind,
            rate=phase.rate,
            min_rate=phase.min_rate,
            max_rate=phase.max_rate,
            max_duration=phase.duration_seconds,
            expected_statuses=phase.expected_statuses,
            drift_severity=args.drift_severity,
        )
        exit_code = max(exit_code, code)
        if args.phase_gap > 0 and phase is not phases[-1]:
            print(f"[demo] pausa entre tramos: {args.phase_gap:.0f}s")
            time.sleep(args.phase_gap)
    return exit_code


# parametros de cada escenario simple, resueltos sobre los flags de la CLI
_SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {"kind": "healthy"},
    "burst": {"kind": "healthy", "min_rate": 8.0, "max_rate": 12.0},
    "churn-spike": {"kind": "churn_spike"},
    "drift": {"kind": "drift"},
    "errors": {"kind": "invalid", "expected_statuses": (422,)},
}


def _run_scenario(
    client: httpx.Client,
    url: str,
    df: pd.DataFrame,
    rng: random.Random,
    args: argparse.Namespace,
) -> int:
    # despacha el escenario pedido; demo tiene su propia secuencia
    if args.scenario == "demo":
        return _run_demo(client, url, df, rng, args)

    spec = _SCENARIOS[args.scenario]
    return _run_phase(
        client,
        url,
        df,
        rng,
        label=args.scenario,
        kind=spec["kind"],
        rate=args.rate,
        min_rate=spec.get("min_rate", args.min_rate),
        max_rate=spec.get("max_rate", args.max_rate),
        max_duration=args.duration,
        expected_statuses=spec.get("expected_statuses", (200,)),
        drift_severity=args.drift_severity,
    )


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    # carga el CSV de muestreo con validaciones de precondicion
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = set(_ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Al CSV le faltan columnas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El CSV no tiene filas")
    return df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera trafico para el monitoreo")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL base de la API")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV de muestreo")
    parser.add_argument("--seed", type=int, default=7, help="semilla de muestreo")

    # modo batch original
    parser.add_argument("--healthy", type=int, default=300, help="requests sanos en modo batch")
    parser.add_argument("--drift", type=int, default=200, help="requests con drift en modo batch")

    # modo loop de fondo
    parser.add_argument("--loop", action="store_true", help="trafico sano de fondo hasta Ctrl+C")
    parser.add_argument("--rate", type=float, default=2.0, help="requests por segundo")
    parser.add_argument(
        "--min-rate", type=float, default=None, help="rate minimo para tasa variable"
    )
    parser.add_argument(
        "--max-rate", type=float, default=None, help="rate maximo para tasa variable"
    )

    # escenarios controlados
    parser.add_argument(
        "--scenario",
        choices=["baseline", "burst", "churn-spike", "drift", "errors", "demo"],
        default=None,
        help="escenario controlado de monitoreo",
    )
    parser.add_argument(
        "--duration", type=float, default=90.0, help="duracion del escenario en segundos"
    )
    parser.add_argument(
        "--demo-scale", type=float, default=1.0, help="escala las duraciones de la demo"
    )
    parser.add_argument(
        "--phase-gap", type=float, default=5.0, help="pausa entre tramos de la demo"
    )
    parser.add_argument(
        "--drift-severity",
        choices=["low", "medium", "high"],
        default="medium",
        help="intensidad del corrimiento en el escenario drift",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    try:
        df = _load_dataframe(args.csv)
        _validate_rate(args.rate, args.min_rate, args.max_rate)
        if args.duration <= 0:
            raise ValueError("--duration debe ser mayor que 0")
        if args.phase_gap < 0:
            raise ValueError("--phase-gap no puede ser negativo")
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with httpx.Client() as client:
        if not _check_health(client, args.url):
            return 1

        # loop de fondo para monitorear en tiempo real
        if args.loop:
            return _run_phase(
                client,
                args.url,
                df,
                rng,
                label="loop",
                kind="healthy",
                rate=args.rate,
                min_rate=args.min_rate,
                max_rate=args.max_rate,
                max_duration=None,
            )

        # escenarios controlados para la demo y la validacion de alertas
        if args.scenario is not None:
            try:
                return _run_scenario(client, args.url, df, rng, args)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

        # modo batch para generar el registro del reporte de drift
        return _run_batch(client, args.url, df, rng, args.healthy, args.drift, args.drift_severity)


if __name__ == "__main__":
    raise SystemExit(main())
