"""Generador de trafico con drift para el reporte de Evidently.
Perfil corrido: clientes nuevos, cargos altos, muchos tickets y pagos
atrasados, contrato mensual y region sur dominantes.
Uso: python scripts/generate_drift.py -n 150 --delay 0.2
"""

import argparse
import random
import time
from pathlib import Path

import httpx
import pandas as pd

API_URL = "http://localhost:8000/predict"

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = ROOT / "data" / "raw" / "churn_sintetico.csv"

REFERENCE = pd.read_csv(REFERENCE_CSV)


def build_drifted_payload() -> dict:
    # mismas claves y rangos validos que el trafico normal, distribuciones corridas
    row = REFERENCE.sample(n=1).iloc[0]

    payload = {
        "tenure_months": int(row["tenure_months"]),
        "monthly_charge": float(row["monthly_charge"]),
        "total_charges": float(row["total_charges"]),
        "support_tickets": int(row["support_tickets"]),
        "late_payments": int(row["late_payments"]),
        "avg_monthly_usage_gb": float(row["avg_monthly_usage_gb"]),
        "contract_type": str(row["contract_type"]),
        "payment_method": str(row["payment_method"]),
        "internet_service": str(row["internet_service"]),
        "has_streaming": int(row["has_streaming"]),
        "has_security_pack": int(row["has_security_pack"]),
        "num_products": int(row["num_products"]),
        "region": str(row["region"]),
        "customer_age": int(row["customer_age"]),
        "is_promo": int(row["is_promo"]),
    }

    payload["tenure_months"] = random.randint(1, 8)
    payload["monthly_charge"] = round(random.uniform(100.0, 127.0), 2)
    payload["support_tickets"] = random.randint(5, 8)
    payload["late_payments"] = random.randint(3, 5)
    payload["contract_type"] = random.choices(
        ["mensual", "anual", "bianual"],
        weights=[8, 1, 1],
    )[0]

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    ok = errors = 0

    with httpx.Client(timeout=5.0) as client:
        for i in range(1, args.n + 1):
            try:
                response = client.post(API_URL, json=build_drifted_payload())
                if response.status_code == 200:
                    ok += 1
                else:
                    errors += 1
            except httpx.HTTPError:
                errors += 1

            if i % 25 == 0:
                print(f"{i}/{args.n} enviados")

            time.sleep(args.delay)

    print(f"fin: {ok} inferencias registradas, {errors} fallidas")


if __name__ == "__main__":
    main()
