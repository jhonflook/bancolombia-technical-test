"""Evaluación de clasificadores de débito recurrente con validación cruzada estratificada."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


@dataclass
class CVResults:
    """Resultados de validación cruzada estratificada para clasificación.

    Attributes
    ----------
    fold_results : pd.DataFrame
        Métricas por fold: auc_roc, ks_stat, f1_05.
    mean_auc : float
        AUC-ROC medio a través de los folds (métrica de optimización).
    std_auc : float
        Desviación estándar del AUC-ROC.
    mean_ks : float
        KS medio a través de los folds.
    mean_f1 : float
        F1 @ 0.5 medio a través de los folds.
    """

    fold_results: pd.DataFrame
    mean_auc: float
    std_auc: float
    mean_ks: float
    mean_f1: float


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Calcular estadístico KS y umbral donde se maximiza la separación.

    El KS mide la máxima diferencia entre TPR y FPR a lo largo de todos
    los umbrales. Es el criterio operativo de cobranza para este proyecto.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas binarias reales.
    y_prob : np.ndarray
        Probabilidades predichas para la clase 1.

    Returns
    -------
    tuple[float, float]
        (ks_statistic, threshold_at_max_ks)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    best_idx = int(np.argmax(ks_values))
    return float(ks_values[best_idx]), float(thresholds[best_idx])


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    partition: str = "eval",
) -> dict[str, float]:
    """Calcular métricas de clasificación binaria alineadas con S8.

    Calcula AUC-ROC, KS, Precision/Recall/F1 a umbral fijo (0.5) y umbral
    óptimo KS, y AUC-PR. Todos los valores se devuelven como un diccionario
    prefijado con el nombre de la partición.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Etiquetas binarias reales.
    y_prob : np.ndarray
        Probabilidades predichas para la clase 1.
    partition : str
        Prefijo para las claves del diccionario (ej. 'train', 'test', 'oot').

    Returns
    -------
    dict[str, float]
        Diccionario con métricas prefijadas por partición.
    """
    y_true = np.asarray(y_true)

    auc_roc = roc_auc_score(y_true, y_prob)
    ks_stat, ks_thr = compute_ks(y_true, y_prob)

    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_ks = (y_prob >= ks_thr).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = float(auc(rec_curve, prec_curve))

    return {
        f"{partition}_auc_roc":      round(float(auc_roc), 4),
        f"{partition}_ks":           round(float(ks_stat), 4),
        f"{partition}_ks_threshold": round(float(ks_thr), 4),
        f"{partition}_precision_05": round(float(precision_score(y_true, y_pred_05, zero_division=0)), 4),
        f"{partition}_recall_05":    round(float(recall_score(y_true, y_pred_05, zero_division=0)), 4),
        f"{partition}_f1_05":        round(float(f1_score(y_true, y_pred_05, zero_division=0)), 4),
        f"{partition}_precision_ks": round(float(precision_score(y_true, y_pred_ks, zero_division=0)), 4),
        f"{partition}_recall_ks":    round(float(recall_score(y_true, y_pred_ks, zero_division=0)), 4),
        f"{partition}_f1_ks":        round(float(f1_score(y_true, y_pred_ks, zero_division=0)), 4),
        f"{partition}_auc_pr":       round(float(auc_pr), 4),
        f"{partition}_n_samples":    float(len(y_true)),
        f"{partition}_n_positive":   float(int(y_true.sum())),
        f"{partition}_prevalence":   round(float(y_true.mean()), 4),
    }


def evaluate_classifier_cv(
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    params: dict[str, Any],
    n_splits: int = 5,
    scale_pos_weight: float = 1.0,
) -> CVResults:
    """Evaluar un clasificador con validación cruzada estratificada.

    Usa StratifiedKFold para mantener la proporción de clases en cada fold,
    esencial dado el desbalance 3.7:1 (S8).

    Parameters
    ----------
    model_class : type
        Clase del clasificador (subclase de BaseDebitClassifier).
    X : pd.DataFrame
        Features completos de entrenamiento.
    y : pd.Series
        Target binario de entrenamiento.
    selected_features : list[str]
        Features a usar en el ajuste.
    params : dict[str, Any]
        Hiperparámetros del modelo.
    n_splits : int
        Número de folds estratificados.
    scale_pos_weight : float
        Cociente n_neg/n_pos para corrección de desbalance (S8).

    Returns
    -------
    CVResults
        Resultados agregados de la validación cruzada.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = model_class()
        clf.fit(
            X_tr,
            y_tr,
            selected_features=selected_features,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        y_prob = clf.predict_proba(X_val)

        ks_stat, _ = compute_ks(y_val.values, y_prob)
        f1 = f1_score(y_val, (y_prob >= 0.5).astype(int), zero_division=0)

        fold_results.append({
            "fold":    fold,
            "auc_roc": float(roc_auc_score(y_val, y_prob)),
            "ks_stat": float(ks_stat),
            "f1_05":   float(f1),
        })

    results_df = pd.DataFrame(fold_results)

    return CVResults(
        fold_results=results_df,
        mean_auc=float(results_df["auc_roc"].mean()),
        std_auc=float(results_df["auc_roc"].std()),
        mean_ks=float(results_df["ks_stat"].mean()),
        mean_f1=float(results_df["f1_05"].mean()),
    )
