"""Generador de trafico para la demo de monitoreo.
 Envia clientes sinteticos validos contra la API y payloads
invalidos para que el panel de tasa de error tenga señal.
"""
# Uso: python scripts/generate_traffic.py -n 200 --delay 0.25

import argparse
import random
import time

import httpx

API_URL = "http://localhost:8000/predict"
INVALID_EVERY = 10  # 1 de cada 10 requests va vacio y cosecha un 422

CONTRACTS = ["mensual", "anual", "bianual"]
PAYMENTS = ["credito", "debito", "efectivo", "transferencia"]
INTERNET = ["cable", "fibra", "movil", "ninguno"]
REGIONS = ["centro", "norte", "oeste", "sur"]


def build_payload() -> dict:
    # cliente sintetico dentro de los rangos observados en el dataset de entrenamiento
    tenure = random.randint(1, 72)
    monthly = round(random.uniform(20.0, 120.0), 2)
    # acumulado coherente por construccion con la antiguedad y el cargo mensual
    total = round(min(max(tenure * monthly * random.uniform(0.85, 1.15), 50.0), 9000.0), 2)
    return {
        "tenure_months": tenure,
        "monthly_charge": monthly,
        "total_charges": total,
        "support_tickets": random.randint(0, 4),
        "late_payments": random.randint(0, 2),
        "avg_monthly_usage_gb": round(random.uniform(20.0, 200.0), 2),
        "contract_type": random.choice(CONTRACTS),
        "payment_method": random.choice(PAYMENTS),
        "internet_service": random.choice(INTERNET),
        "has_streaming": random.randint(0, 1),
        "has_security_pack": random.randint(0, 1),
        "num_products": random.randint(1, 4),
        "region": random.choice(REGIONS),
        "customer_age": random.randint(18, 78),
        "is_promo": random.randint(0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trafico sintetico contra la API de churn")
    parser.add_argument("-n", type=int, default=200, help="cantidad de requests")
    parser.add_argument("--delay", type=float, default=0.25, help="pausa entre requests (s)")
    args = parser.parse_args()

    ok = invalid = errors = 0
    with httpx.Client(timeout=5.0) as client:
        for i in range(1, args.n + 1):
            body = {} if i % INVALID_EVERY == 0 else build_payload()
            try:
                r = client.post(API_URL, json=body)
                if r.status_code == 200:
                    ok += 1
                elif r.status_code == 422:
                    invalid += 1
                else:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            if i % 25 == 0:
                print(f"{i}/{args.n} enviados")
            time.sleep(args.delay)

    print(f"fin: {ok} ok, {invalid} rechazados (422), {errors} errores de red")


if __name__ == "__main__":
    main()
