#evaluate.py

'''
evaluacion final del modelo sobre el conjunto de test.

observaciones:
*esto corre UNA sola vez, sobre datos nunca vistos en train ni val

*no se usa para selección de modelos -> eso se hizo en src/train.py con el
conjunto de validación.

'''

import json
from pathlib import Path
from typing import Final

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import load_raw_data, split_data


MODEL_PATH: Final[Path] = Path("models/churn_model_v1.joblib")
REPORTS_DIR: Final[Path] = Path("reports")
METRICS_PATH: Final[Path] = REPORTS_DIR / "test_metrics.json"

# cargar la data del JSON generado en el train

def _load_winner_info() -> tuple[str, float]:
    """Lee el ganador (mayor f1_val) y su threshold del train_metrics.json."""
    metrics_path = REPORTS_DIR / "train_metrics.json"
    with open(metrics_path) as f:
        all_results = json.load(f)
    winner = max(all_results, key=lambda m: all_results[m]["f1_val"])
    return winner, float(all_results[winner]["threshold"])

def _load_optimal_threshold(model_name: str = "LogisticRegression") -> float:
    metrics_path = REPORTS_DIR / "train_metrics.json"
    with open(metrics_path) as f:
        all_results = json.load(f)
    return float(all_results[model_name]["threshold"])

#eval 


def evaluate_on_test(
    model_path: Path = MODEL_PATH,
    threshold: float | None = None,
    model_name: str | None = None,
) -> dict[str, float]:
    # cargo el modelo y el threshold ganador
    if threshold is None or model_name is None:
        winner, winner_threshold = _load_winner_info()
        if threshold is None:
            threshold = winner_threshold
        if model_name is None:
            model_name = winner
        print(f"  Modelo ganador detectado: {model_name}")

    # prueba el modelo final sobre el conjunto de test...
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            "Corré src/train.py primero."
        )
    
    REPORTS_DIR.mkdir(exist_ok=True)

    #crga todo
    pipeline: Pipeline = joblib.load(model_path)
    df = load_raw_data()
    _, _, X_test, _, _, y_test = split_data(df)

    # predicciones
    y_proba: np.ndarray = pipeline.predict_proba(X_test)[:, 1]
    y_pred: np.ndarray = (y_proba >= threshold).astype(int)

    #metricas
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report: dict = classification_report(y_test, y_pred, output_dict=True)

    print("=" * 55)
    print(f"EVALUACIÓN FINAL — TEST SET (n={len(y_test)})")
    print("=" * 55)
    print(f"  Threshold usado: {threshold}")
    print(f"  F1-score:        {f1:.4f}")
    print(f"  ROC-AUC:         {roc_auc:.4f}")
    print(f"  PR-AUC:          {pr_auc:.4f}")
    print(f"  Precision (1):   {report['1']['precision']:.4f}")
    print(f"  Recall    (1):   {report['1']['recall']:.4f}")
    print("=" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=["No churn", "Churn"],
    ))

    # test basico
    assert 0.0 <= f1 <= 1.0, f"F1 fuera de rango: {f1}"
    assert 0.0 <= roc_auc <= 1.0, f"ROC-AUC fuera de rango: {roc_auc}"
    assert len(y_pred) == len(y_test), "Cantidad de predicciones no coincide con test"

    metrics = {
        "model": str(model_path),
        "model_name": model_name,
        "threshold": threshold,
        "test_size": len(y_test),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_class1": round(report["1"]["precision"], 4),
        "recall_class1": round(report["1"]["recall"], 4),
        "support_class1": int(report["1"]["support"]),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n metricas guardadas en {METRICS_PATH}")

    return metrics


# visualizaciones

def plot_evaluation(
    model_path: Path = MODEL_PATH,
    threshold: float | None = None,
    model_name: str | None = None,
) -> None:
    #Genera y guarda las tres visualizaciones de evaluación.
    '''
        Plots generados:
        - confusion matrix con threshold calibrado
        - ROC curve con área bajo la curva
        - Precision-Recall curve con area bajo la curva
    '''
        
    if threshold is None or model_name is None:
        winner, winner_threshold = _load_winner_info()
        if threshold is None:
            threshold = winner_threshold
        if model_name is None:
            model_name = winner

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    pipeline: Pipeline = joblib.load(model_path)
    df = load_raw_data()
    _, _, X_test, _, _, y_test = split_data(df)

    y_proba: np.ndarray = pipeline.predict_proba(X_test)[:, 1]
    y_pred: np.ndarray = (y_proba >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # graf. 1: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No churn", "Churn"],
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
    )
    axes[0].set_title(
        f"Confusion Matrix - Test set\n(threshold = {threshold})",
        fontsize=11, fontweight="bold",
    )

    # graf. 2: ROC Curve
    RocCurveDisplay.from_predictions(
        y_test, y_proba,
        ax=axes[1],
        color="steelblue",
        name=model_name,
    )
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.5)")
    axes[1].set_title("ROC Curve - Test set", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    # graf. 3: Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba,
        ax=axes[2],
        color="coral",
        name=model_name,
    )
    # baseline: precision = proporcion de positivos
    baseline = float(y_test.mean())
    axes[2].axhline(
        baseline, color="gray", linestyle="--",
        linewidth=0.8, label=f"Baseline (p={baseline:.2f})",
    )
    # marca el threshold elegido
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    axes[2].scatter(
        recalls[idx], precisions[idx],
        color="black", s=80, zorder=5,
        label=f"Threshold = {threshold}",
    )
    axes[2].set_title(
        "Precision-Recall Curve - Test set",
        fontsize=11, fontweight="bold",
    )
    axes[2].legend(fontsize=9)

    plt.suptitle(
        f"Evaluación final - {model_name} | Test set (n=1000)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    output_path = REPORTS_DIR / "eda_07_evaluation_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"grafico guardado en {output_path}")


#entry point
if __name__ == "__main__":
    winner_name, winner_threshold = _load_winner_info()
    metrics = evaluate_on_test(threshold=winner_threshold, model_name=winner_name)
    plot_evaluation(threshold=winner_threshold, model_name=winner_name)