"""Reporte de data drift con Evidently.
Compara la distribucion de las inferencias registradas en produccion
(monitoring/data/inferences.jsonl) contra el dataset de entrenamiento y
genera un HTML con el resultado por feature.
"""
# Uso: python scripts/report_drift.py

import json
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = ROOT / "data" / "raw" / "churn_sintetico.csv"
CURRENT_JSONL = ROOT / "monitoring" / "data" / "inferences.jsonl"
OUTPUT_HTML = ROOT / "reports" / "drift_report.html"
MIN_ROWS = 40  # debajo de esto los tests estadisticos del preset no son confiables

NUMERIC = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "has_streaming",
    "has_security_pack",
    "num_products",
    "customer_age",
    "is_promo",
]
CATEGORICAL = ["contract_type", "payment_method", "internet_service", "region"]
FEATURES = NUMERIC + CATEGORICAL


def load_current() -> pd.DataFrame:
    rows = []
    for line in CURRENT_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # tolera registro plano o con las features anidadas bajo "features"
        rows.append({**record.get("features", {}), **record})
    return pd.DataFrame(rows)


def main() -> int:
    if not REFERENCE_CSV.exists():
        print(f"falta la referencia {REFERENCE_CSV}, correr make pull primero")
        return 1
    if not CURRENT_JSONL.exists():
        print(f"no hay inferencias registradas en {CURRENT_JSONL}")
        return 1

    reference = pd.read_csv(REFERENCE_CSV)[FEATURES]
    current = load_current()

    missing = [c for c in FEATURES if c not in current.columns]
    if missing:
        print(f"faltan columnas en el JSONL: {missing}")
        return 1
    current = current[FEATURES].copy()

    if len(current) < MIN_ROWS:
        print(f"solo {len(current)} inferencias, minimo {MIN_ROWS} para un reporte confiable")
        return 1

    # normalizacion de tipos antes de pasar los datos a Evidently
    for col in NUMERIC:
        current[col] = pd.to_numeric(current[col], errors="coerce")
    for col in CATEGORICAL:
        reference[col] = reference[col].astype("string")
        current[col] = current[col].astype("string")

    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)
    current_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()], include_tests=True)
    snapshot = report.run(current_ds, reference_ds)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(OUTPUT_HTML))
    print(f"reporte generado: {OUTPUT_HTML}")
    print(f"ventana actual: {len(current)} inferencias vs {len(reference)} filas de referencia")
    return 0


if __name__ == "__main__":
    sys.exit(main())
