# data.py

"""
Carga de datos y split estratificado para el modelo de churn

Split: 64% train / 16% val / 20% test (estratificado por target)
El split es determinístico: RANDOM_SEED fijo garantiza reproducibilidad

"""

from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import TARGET, add_derived_features

RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.20  # 20% para test
VAL_SIZE: Final[float] = 0.20  # 20% del 80% restante = 16% del total
DATA_PATH: Final[Path] = Path("data/raw/churn_sintetico.csv")

# Tolerancia máxima de desviación en proporción de churn entre splits
STRATIFICATION_TOLERANCE: Final[float] = 0.02


# carga


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    # Carga el CSV original y aplica feature engineering.

    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado en: {path}")

    df = pd.read_csv(path)

    # Precondiciones sobre el CSV
    if df.shape[0] == 0:
        raise ValueError("El CSV está vacío")
    if TARGET not in df.columns:
        raise ValueError(f"Columna target '{TARGET}' no encontrada en el CSV")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"Target '{TARGET}' debe ser binario (0/1)")

    df = add_derived_features(df)
    return df


# split


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split estratificado en train / val / test (64% / 16% / 20%).

    La estratificación garantiza que la proporción de churn se mantiene
    en los tres subconjuntos dentro de una tolerancia de ±2%.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Columna '{TARGET}' no encontrada")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    churn_rate_global = y.mean()

    # split 1: train+val (80%) vs test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # split 2: train (64%) vs val (16%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    total_split = len(X_train) + len(X_val) + len(X_test)
    assert total_split == len(df), f"Se perdieron filas en el split: {len(df)} → {total_split}"

    # estratificación correcta?
    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        deviation = abs(y_split.mean() - churn_rate_global)
        assert deviation < STRATIFICATION_TOLERANCE, (
            f"Estratificación falló en {name}: "
            f"churn rate {y_split.mean():.3f} vs global {churn_rate_global:.3f} "
            f"(desviación {deviation:.3f} > tolerancia {STRATIFICATION_TOLERANCE})"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


# informe de splits


def report_splits(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    # Imprime un resumen de tamaños y proporcion de churn por split
    total = len(y_train) + len(y_val) + len(y_test)
    print("=" * 50)
    print("RESUMEN DE SPLITS")
    print("=" * 50)
    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pct = len(y_split) / total * 100
        print(
            f"  {name:<6}: {len(y_split):>4} filas ({pct:.0f}%) | churn rate: {y_split.mean():.3f}"
        )
    print("=" * 50)


# entry point
if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    report_splits(y_train, y_val, y_test)
    print("src/data.py: OK")
