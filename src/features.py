'''

pipeline de preprocesamiento y feature engineering para el modelo de churn

observaciones:

*total_charges se elimina: redundancia estructural alta con tenure × monthly_charge
*charges_per_month reemplaza a total_charges (evaluar vs monthly_charge post-entrenamiento)
*tickets_per_year con clip(lower=6) para evitar inflación en clientes de baja antigüedad
*SimpleImputer incluido aunque el dataset no tenga nulos: robustez ante drift de calidad

'''

from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED: Final[int] = 42
TARGET: Final[str] = "churn"

#minimo de meses para anualizar tickets sin inflar artificialmente la tasa
# ---> Clientes con menos de 6 meses se tratan como si tuvieran 6<---
TENURE_MIN_FOR_ANNUALIZATION: Final[int] = 6

NUMERIC_FEATURES: Final[list[str]] = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "has_streaming",
    "has_security_pack",
    "num_products",
    "customer_age",
    "is_promo",
    # Features derivadas — creadas por add_derived_features() antes del pipeline
    "charges_per_month",
    "tickets_per_year",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
]

# total_charges se excluye explícitamente: redundancia estructural con tenure × monthly.
FEATURES_TO_DROP: Final[list[str]] = ["total_charges"]


# feature engineering 


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    agrega features derivadas al DataFrame y retorna una copia.

    features creadas:
    - charges_per_month: gasto histórico promedio mensual real.
      Nota: puede correlacionar fuertemente con monthly_charge dado el comportamiento
      algebraico del dataset sintético. Se evaluará su aporte post-entrenamiento
      via importancia de features.
    - tickets_per_year: tickets de soporte anualizados, con protección para
      clientes de baja antigüedad (clip lower=6 meses) para evitar inflacion
      artificial en clientes recientes.

    '''
    required_cols = {"total_charges", "tenure_months", "monthly_charge", "support_tickets"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes para feature engineering: {missing}")

    df = df.copy()

    # charges_per_month: gasto real histórico promedio
    # Para tenure=0 usamos monthly_charge como fallback (cliente sin historial)
    df["charges_per_month"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"],
        df["monthly_charge"],
    )

    # tickets_per_year: tasa anualizada de tickets de soporte
    # clip(lower=6): clientes con < 6 meses se tratan como 6 meses
    # para evitar ratios artificialmente altos (ej: 3 tickets en 1 mes → 36/año)
    tenure_clipped = df["tenure_months"].clip(lower=TENURE_MIN_FOR_ANNUALIZATION)
    df["tickets_per_year"] = df["support_tickets"] / (tenure_clipped / 12)

    #postcondiciones
    assert not df["charges_per_month"].isnull().any(), "NaN en charges_per_month"
    assert not df["tickets_per_year"].isnull().any(), "NaN en tickets_per_year"
    assert (df["charges_per_month"] >= 0).all(), "charges_per_month con valores negativos"
    assert (df["tickets_per_year"] >= 0).all(), "tickets_per_year con valores negativos"

    return df


#preprocessor


def build_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer de preprocesamiento.

    Pipeline numérico:
        SimpleImputer(median) → StandardScaler
    Pipeline categórico:
        SimpleImputer(most_frequent) → OneHotEncoder(drop='first')

    El SimpleImputer se incluye aunque el dataset actual no tenga nulos:
    garantiza que el pipeline no falle ante datos con nulos en producción
    (drift de calidad de datos).

    remainder='drop' descarta total_charges y cualquier otra columna no listada.

    Returns:
        ColumnTransformer listo para fit/transform.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",        # evita multicolinealidad perfecta en OHE
                    sparse_output=False, # array denso para compatibilidad con sklearn
                    handle_unknown="error",  # falla ruidosamente ante categoría nueva
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # descarta total_charges y columnas no listadas
    )

    return preprocessor


# features


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extrae los nombres de features post-transformación.

    Útil para interpretabilidad: importancia de features en RF/HistGBM.
    Solo válido después de llamar a preprocessor.fit().

    Args:
        preprocessor: ColumnTransformer ya fiteado.

    Returns:
        Lista con nombres de todas las features en el orden en que
        el preprocessor las genera.

    Raises:
        RuntimeError: si el preprocessor no fue fiteado previamente.
    """
    try:
        cat_encoder: OneHotEncoder = (
            preprocessor.named_transformers_["cat"].named_steps["encoder"]
        )
    except AttributeError as exc:
        raise RuntimeError(
            "El preprocessor no fue fiteado. Llamá a preprocessor.fit() primero."
        ) from exc

    cat_names: list[str] = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_names


#smoke test 


def _smoke_test() -> None:
    #verifica que el pipeline de features funciona end-to-end con data minima
    sample = pd.DataFrame(
        {
            "tenure_months": [24, 0, 3],
            "monthly_charge": [65.0, 30.0, 80.0],
            "total_charges": [1560.0, 0.0, 240.0],
            "support_tickets": [2, 0, 5],
            "late_payments": [1, 0, 2],
            "avg_monthly_usage_gb": [120.0, 50.0, 200.0],
            "has_streaming": [1, 0, 1],
            "has_security_pack": [0, 1, 0],
            "num_products": [2, 1, 3],
            "customer_age": [35, 22, 60],
            "is_promo": [0, 1, 0],
            "contract_type": ["mensual", "anual", "bianual"],
            "payment_method": ["debito", "credito", "efectivo"],
            "internet_service": ["fibra", "cable", "movil"],
            "region": ["centro", "norte", "sur"],
            "churn": [0, 1, 0],
        }
    )

    # test add_derived_features
    df_feat = add_derived_features(sample)
    assert "charges_per_month" in df_feat.columns
    assert "tickets_per_year" in df_feat.columns

    # tenure=0 → fallback a monthly_charge
    assert df_feat.loc[1, "charges_per_month"] == 30.0

    # tenure=3 (< 6) → clip a 6 para anualizar
    # tickets_per_year = 5 / (6/12) = 10.0
    from math import isclose
    assert isclose(df_feat.loc[2, "tickets_per_year"], 10.0, rel_tol=1e-6), \
    f"tickets_per_year esperado 10.0, got {df_feat.loc[2, 'tickets_per_year']}"

    # test preprocessor
    X = df_feat.drop(columns=["churn"])
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor)

    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] == len(feature_names)
    assert not np.isnan(X_transformed).any()

    print("    Smoke test de features.py: OK")
    print(f"   Features totales post-transformación: {X_transformed.shape[1]}")
    print(f"   Nombres: {feature_names}")


if __name__ == "__main__":
    _smoke_test()