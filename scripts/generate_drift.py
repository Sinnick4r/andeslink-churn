"""Generador de trafico con drift para el reporte de Evidently.
Perfil corrido: clientes nuevos, cargos altos, muchos tickets y pagos
atrasados, contrato mensual y region sur dominantes.
Uso: python scripts/generate_drift.py -n 150 --delay 0.2
"""

import argparse
import random
import time

import httpx

API_URL = "http://localhost:8000/predict"

# poblaciones con pesos: la primera opcion domina y genera drift categorico visible
CONTRACTS_W = (["mensual", "anual", "bianual"], [8, 1, 1])
INTERNET_W = (["movil", "cable", "fibra", "ninguno"], [6, 2, 1, 1])
REGIONS_W = (["sur", "centro", "norte", "oeste"], [7, 1, 1, 1])
PAYMENTS = ["credito", "debito", "efectivo", "transferencia"]


def build_drifted_payload() -> dict:
    # mismas claves y rangos validos que el trafico normal, distribuciones corridas
    tenure = random.randint(1, 8)
    monthly = round(random.uniform(100.0, 127.0), 2)
    total = round(min(max(tenure * monthly * random.uniform(0.85, 1.15), 50.0), 9000.0), 2)
    return {
        "tenure_months": tenure,
        "monthly_charge": monthly,
        "total_charges": total,
        "support_tickets": random.randint(5, 8),
        "late_payments": random.randint(3, 5),
        "avg_monthly_usage_gb": round(random.uniform(5.0, 40.0), 2),
        "contract_type": random.choices(*CONTRACTS_W)[0],
        "payment_method": random.choice(PAYMENTS),
        "internet_service": random.choices(*INTERNET_W)[0],
        "has_streaming": random.choices([0, 1], weights=[8, 2])[0],
        "has_security_pack": random.choices([0, 1], weights=[9, 1])[0],
        "num_products": 1,
        "region": random.choices(*REGIONS_W)[0],
        "customer_age": random.randint(18, 30),
        "is_promo": random.choices([1, 0], weights=[8, 2])[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trafico con drift contra la API de churn")
    parser.add_argument("-n", type=int, default=150, help="cantidad de requests")
    parser.add_argument("--delay", type=float, default=0.2, help="pausa entre requests (s)")
    args = parser.parse_args()

    ok = errors = 0
    with httpx.Client(timeout=5.0) as client:
        for i in range(1, args.n + 1):
            try:
                r = client.post(API_URL, json=build_drifted_payload())
                if r.status_code == 200:
                    ok += 1
                else:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            if i % 25 == 0:
                print(f"{i}/{args.n} enviados")
            time.sleep(args.delay)

    print(f"fin: {ok} inferencias con drift registradas, {errors} fallidas")


if __name__ == "__main__":
    main()
