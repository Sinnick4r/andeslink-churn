# train.py
# Entrenamiento de modelos con tracking en MLflow.
"""
Modelos comparados:
- LogisticRegression: baseline lineal con class_weight='balanced'
- RandomForestClassifier: ensemble de árboles
- HistGradientBoostingClassifier: gradient boosting nativo de sklearn
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import load_raw_data, report_splits, split_data
from src.features import build_preprocessor

RANDOM_SEED: Final[int] = 42
MODEL_DIR: Final[Path] = Path("models")
REPORTS_DIR: Final[Path] = Path("reports")
EXPERIMENT_NAME: Final[str] = "andeslink-churn-e2"
MODEL_FILENAME: Final[str] = "churn_model_v2.joblib"
F1_TOLERANCE = 0.005

# Ante diferencias despreciables, elijo siempre el modelo mas simple
MODEL_PRIORITY = {
    "LogisticRegression": 0,
    "HistGradientBoosting": 1,
    "RandomForest": 2,
}
# Utils


def compute_model_hash(model_path: Path) -> str:
    # SHA-256 del artefacto serializado -> String hexadecimal del hash SHA-256.

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    # Threshold que maximiza F1 sobre la curva precision-recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve retorna len(thresholds) = len(precisions) - 1
    # El ult. punto (recall=0, precision=1) no tiene threshold asociado
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    )
    best_idx = int(np.argmax(f1_scores))
    return round(float(thresholds[best_idx]), 6)


# Config de modelos


def get_model_configs() -> dict[str, dict[str, Any]]:
    # Define los 3 modelos a comparar c
    # Todos usan sklearn puro
    # Return ->> Dict con nombre → {model, params} para logging en MLflow.

    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            ),
            "params": {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "RandomForest",
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            ),
            "params": {
                "model_type": "HistGradientBoosting",
                "max_iter": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_samples_leaf": 20,
            },
        },
    }


# TRAINING


def train_and_evaluate(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict[str, Any]:
    # Entrena un pipeline completo, evalua en validación, devuelve Dict con pipeline, threshold, metricas y reporte.

    # Precondiciones
    assert X_train.shape[0] == y_train.shape[0], "X_train e y_train tienen distinto n"
    assert X_val.shape[0] == y_val.shape[0], "X_val e y_val tienen distinto n"
    assert not X_train.isnull().all().any(), "Columna completamente nula en X_train"

    # pipeline: preprocessor + clasificador
    full_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )

    full_pipeline.fit(X_train, y_train)

    # probabilidades en validacion
    y_proba: np.ndarray = full_pipeline.predict_proba(X_val)[:, 1]

    # Threshold óptimo por F1
    threshold = find_optimal_threshold(y_val.values, y_proba)
    y_pred: np.ndarray = (y_proba >= threshold).astype(int)

    # metricas
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    report: dict[str, Any] = classification_report(y_val, y_pred, output_dict=True)

    # metricas en rango esperado
    assert 0.0 <= f1 <= 1.0, f"F1 fuera de rango: {f1}"
    assert 0.0 <= roc_auc <= 1.0, f"ROC-AUC fuera de rango: {roc_auc}"

    print(
        f"  F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | "
        f"PR-AUC: {pr_auc:.4f} | Threshold: {threshold:.3f}"
    )
    print(f"  Confusion matrix:\n{cm}")

    return {
        "pipeline": full_pipeline,
        "threshold": threshold,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# Training principal


def run_experiment() -> None:
    """
    Ejecuta el pipeleine completo: 3 modelos, MLflow tracking, serialización.

    Side effects:
    - Escribe runs en mlruns/ (MLflow local)
    - Escribe models/churn_model_v2.joblib
    - Escribe reports/train_metrics.json
    """
    # Setup
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Carga y split
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    report_splits(y_train, y_val, y_test)

    # MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_configs = get_model_configs()

    all_results: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    # Lloop de training
    for model_name, config in model_configs.items():
        print(f"\n{'=' * 55}")
        print(f"Entrenando: {model_name}")
        print(f"{'=' * 55}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(config["params"])
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("churn_rate_train", round(float(y_train.mean()), 4))

            results = train_and_evaluate(
                model_name=model_name,
                model=config["model"],
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
            )

            # Log métricas
            mlflow.log_metric("f1_val", results["f1"])
            mlflow.log_metric("roc_auc_val", results["roc_auc"])
            mlflow.log_metric("pr_auc_val", results["pr_auc"])
            mlflow.log_metric("optimal_threshold", results["threshold"])
            mlflow.log_metric(
                "precision_class1",
                results["classification_report"]["1"]["precision"],
            )
            mlflow.log_metric(
                "recall_class1",
                results["classification_report"]["1"]["recall"],
            )

            all_results[model_name] = {
                "f1_val": results["f1"],
                "roc_auc_val": results["roc_auc"],
                "pr_auc_val": results["pr_auc"],
                "threshold": results["threshold"],
            }
            trained_pipelines[model_name] = results["pipeline"]

            # Actualizar mejor modelo
            """
            if results["f1"] > best_f1:
                best_f1 = results["f1"]
                best_model_name = model_name
                best_pipeline = results["pipeline"]
                best_threshold = results["threshold"]
            """
    best_f1_absolute = max(result["f1_val"] for result in all_results.values())

    candidate_names = [
        name
        for name, result in all_results.items()
        if result["f1_val"] >= best_f1_absolute - F1_TOLERANCE
    ]

    best_model_name = sorted(
        candidate_names,
        key=lambda name: (
            MODEL_PRIORITY[name],
            -all_results[name]["roc_auc_val"],
        ),
    )[0]

    best_pipeline = trained_pipelines[best_model_name]
    best_f1 = all_results[best_model_name]["f1_val"]
    best_threshold = all_results[best_model_name]["threshold"]
    selection_rule = (
        f"Seleccion por F1 con tolerancia {F1_TOLERANCE}. "
        "En empate operativo, desempate por simplicidad/interpretabildad "
        "y luego ROC-AUC."
    )
    # serializacion del mejor modelo
    assert best_pipeline is not None, "Ningún modelo fue entrenado correctamente"

    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(best_pipeline, model_path)

    model_hash = compute_model_hash(model_path)

    print(f"\n{'=' * 55}")
    print(f"MODELO GANADOR: {best_model_name}")
    print(f"F1 (val): {best_f1:.4f} | Threshold: {best_threshold:.3f}")
    print(f"Serializado: {model_path}")
    print(f"SHA-256: {model_hash[:24]}...")
    print(f"{'=' * 55}")
    metrics_payload = {
        "selected_model": best_model_name,
        "selection_rule": selection_rule,
        "f1_tolerance": F1_TOLERANCE,
        "best_f1_absolute": best_f1_absolute,
        "candidate_models": candidate_names,
        "selected_threshold": best_threshold,
        "models": all_results,
    }
    # run de resumen en MLflow
    with mlflow.start_run(run_name=f"GANADOR_{best_model_name}"):
        mlflow.log_param("winner_model", best_model_name)
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_param("model_sha256_prefix", model_hash[:16])
        mlflow.log_metric("best_f1_val", best_f1)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        # Log comparativa de los 3 modelos como artifact JSON
        metrics_path = REPORTS_DIR / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

    # smoke test del artefacto serializado
    loaded_pipeline: Pipeline = joblib.load(model_path)
    smoke_input = X_val.iloc[:5]
    smoke_pred = loaded_pipeline.predict(smoke_input)
    smoke_proba = loaded_pipeline.predict_proba(smoke_input)[:, 1]

    assert smoke_pred.shape == (5,), f"Shape inesperado: {smoke_pred.shape}"
    assert all(0.0 <= p <= 1.0 for p in smoke_proba), "Probabilidades fuera de [0,1]"
    assert set(smoke_pred).issubset({0, 1}), "Predicciones fuera de {0,1}"

    print("\n Smoke test del .joblib: OK")
    print(f"   Predicciones: {smoke_pred.tolist()}")
    print(f"   Probabilidades: {[round(p, 4) for p in smoke_proba.tolist()]}")


# Entry point
if __name__ == "__main__":
    run_experiment()
