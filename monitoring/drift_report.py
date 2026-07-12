"""Reporte de data drift y prediction drift con Evidently

Compara la distribucion de las 15 features de entrada entre:
- referencia: el split de train del dataset en sus columnas crudas (las que envia
  el cliente), obtenido con el mismo split determinastico que uso el modelo
- actual: las inferencias registradas en produccion (JSONL del registro)

Genera un HTML interpretable con el drift por feature, el drift del churn predicho
respecto al churn de entrenamiento y un resumen de calidad de datos. Sobre el
churn: en la referencia es el valor real del dataset y en el actual es la clase
predicha, porque en produccion todavia no hay ground truth. Comparar ambas
distribuciones es la forma habitual de vigilar prediction drift sin etiquetas.

Se corre desde el entorno conda del proyecto, que ya incluye Evidently, sin
formar parte del runtime de la API. No forma parte del pipeline DVC porque el
insumo actual es dinamico: depende del trafico de produccion.

Uso:
    python monitoring/drift_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# permite ejecutar el script directamente desde la raiz del repo
# agrega la raiz al path para resolver el import de src sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from evidently import ColumnMapping  # noqa: E402
from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # noqa: E402
from evidently.report import Report  # noqa: E402

from src.data import load_raw_csv, split_data  # noqa: E402

# columnas de entrada particionadas como las espera Evidently
_NUMERICAL = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "num_products",
    "customer_age",
]
_CATEGORICAL = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
    "has_streaming",
    "has_security_pack",
    "is_promo",
]
_FEATURES = _NUMERICAL + _CATEGORICAL
TARGET = "churn"

DEFAULT_REGISTRY = Path("monitoring/data/inferences.jsonl")
DEFAULT_OUTPUT = Path("monitoring/reports/drift_report.html")


def load_reference() -> pd.DataFrame:
    # split de train en columnas crudas, mismas filas que vio el modelo
    df_raw = load_raw_csv()
    x_train, _, _, y_train, _, _ = split_data(df_raw)
    reference = x_train[_FEATURES].copy()
    reference[TARGET] = y_train.to_numpy()
    return reference


def load_current(registry_path: Path) -> pd.DataFrame:
    # inferencias registradas en produccion, una por linea JSONL
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No hay registro de inferencias en {registry_path}. "
            "Genera trafico con monitoring/generate_traffic.py primero."
        )
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError("El registro de inferencias esta vacio")
    current = pd.DataFrame(rows)
    return current[_FEATURES + [TARGET]].copy()


def build_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: Path) -> None:
    column_mapping = ColumnMapping(
        target=TARGET,
        numerical_features=_NUMERICAL,
        categorical_features=_CATEGORICAL,
    )
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera el reporte de drift con Evidently")
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="JSONL de inferencias"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="HTML de salida")
    args = parser.parse_args()

    try:
        reference = load_reference()
        current = load_current(args.registry)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Referencia (train): {len(reference)} filas")
    print(f"Actual (produccion): {len(current)} filas")
    build_report(reference, current, args.output)
    print(f"Reporte generado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
