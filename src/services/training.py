"""Entrenamiento y evaluación del clasificador de débitos recurrentes.

Entrena XGBoost (binario) con corrección por desbalance de clases (scale_pos_weight).
Evalúa en Train, Test y OOT con métricas operativas de cobranza.
Registra experimento, parámetros, métricas y artefactos en MLflow.

Pipeline step 4 (standalone):
    python -m src.services.training \
        --data-dir /app/data/artifacts \
        --mlflow-uri http://mlflow:5000

Métricas principales:
  - AUC-ROC:  discriminación global del modelo (métrica primaria, S8)
  - KS:       separación máxima entre distribuciones clase 0 y clase 1
  - Precision / Recall / F1 @ umbral 0.5 y @ umbral KS
  - AUC-PR:   área bajo la curva Precision-Recall (relevante con desbalance)

Supuestos:
  S7: particiones Train/Test/OOT ya definidas en feature_engineering step.
  S8: scale_pos_weight = n_neg/n_pos corrige desbalance 3.7:1.
  S9: pago_debito_* incluidos (ventanas históricas, sin leakage).
"""

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "debit_recurrence_classifier"
TARGET_COL = "var_rta"
JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]

# Hiperparámetros base del clasificador (ajustables vía Optuna en fase posterior)
XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "early_stopping_rounds": 40,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ─── Métricas ────────────────────────────────────────────────────────────────

@dataclass
class PartitionMetrics:
    """Métricas de evaluación para una partición (train / test / OOT)."""

    partition: str
    n_samples: int
    n_positive: int
    prevalence: float
    auc_roc: float
    ks_stat: float
    ks_threshold: float
    precision_05: float
    recall_05: float
    f1_05: float
    precision_ks: float
    recall_ks: float
    f1_ks: float
    auc_pr: float
    report: str = field(default="", repr=False)


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Calcular estadístico KS y el umbral donde se maximiza la separación.

    El KS mide la máxima diferencia entre la tasa de verdaderos positivos
    y la tasa de falsos positivos a lo largo de todos los umbrales posibles.

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


def evaluate_partition(
    model: XGBClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    partition: str,
) -> PartitionMetrics:
    """
    Evaluar el clasificador sobre una partición de datos.

    Parameters
    ----------
    model : XGBClassifier
        Modelo entrenado con predict_proba().
    df : pd.DataFrame
        Partición de datos con features y target.
    feature_cols : list[str]
        Columnas de features seleccionadas en el step de feature engineering.
    partition : str
        Nombre de la partición: 'train', 'test' o 'oot'.

    Returns
    -------
    PartitionMetrics
        Contenedor con todas las métricas calculadas.
    """
    X = df[feature_cols].fillna(0.0)
    y = df[TARGET_COL].astype(int)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, y_prob)
    ks, ks_thresh = compute_ks(y.values, y_prob)

    y_pred_ks = (y_prob >= ks_thresh).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y, y_prob)
    auc_pr = float(auc(rec_curve, prec_curve))

    prevalence = float(y.mean())
    report = classification_report(y, y_pred_05, digits=3)

    m = PartitionMetrics(
        partition=partition,
        n_samples=len(df),
        n_positive=int(y.sum()),
        prevalence=round(prevalence, 4),
        auc_roc=round(auc_roc, 4),
        ks_stat=round(ks, 4),
        ks_threshold=round(ks_thresh, 4),
        precision_05=round(precision_score(y, y_pred_05, zero_division=0), 4),
        recall_05=round(recall_score(y, y_pred_05, zero_division=0), 4),
        f1_05=round(f1_score(y, y_pred_05, zero_division=0), 4),
        precision_ks=round(precision_score(y, y_pred_ks, zero_division=0), 4),
        recall_ks=round(recall_score(y, y_pred_ks, zero_division=0), 4),
        f1_ks=round(f1_score(y, y_pred_ks, zero_division=0), 4),
        auc_pr=auc_pr,
        report=report,
    )

    logger.info(
        "[%s] n=%d | AUC=%.4f | KS=%.4f (thr=%.3f) | "
        "P@0.5=%.3f | R@0.5=%.3f | F1@0.5=%.3f | AUC-PR=%.4f",
        partition, m.n_samples, m.auc_roc, m.ks_stat, m.ks_threshold,
        m.precision_05, m.recall_05, m.f1_05, m.auc_pr,
    )
    return m


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_xgboost(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> XGBClassifier:
    """
    Entrenar XGBoost con corrección por desbalance de clases.

    Parameters
    ----------
    df_train : pd.DataFrame
        Datos de entrenamiento con features y target.
    feature_cols : list[str]
        Columnas de features a usar.
    params : dict or None
        Hiperparámetros del modelo. Si es None usa XGBOOST_PARAMS.

    Returns
    -------
    XGBClassifier
        Modelo entrenado.

    Notes
    -----
    S8: scale_pos_weight = n_neg/n_pos compensa el desbalance 3.7:1.
        Early stopping usa un 15% del train como conjunto de validación
        interna para evitar overfitting sin tocar la partición Test.
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()

    X_all = df_train[feature_cols].fillna(0.0)
    y_all = df_train[TARGET_COL].astype(int)

    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    params["scale_pos_weight"] = float(n_neg / n_pos) if n_pos > 0 else 1.0

    logger.info(
        "Entrenando XGBoost | n=%d | pos=%d (%.1f%%) | scale_pos_weight=%.3f",
        len(df_train), n_pos, 100.0 * n_pos / len(df_train), params["scale_pos_weight"],
    )

    # 15% final del train como validación para early stopping (orden temporal)
    split_idx = int(len(X_all) * 0.85)
    X_tr, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_tr, y_val = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    model = XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(
        "Entrenamiento completo | best_iteration=%d | best_score=%.4f",
        model.best_iteration, model.best_score,
    )
    return model


# ─── Logging MLflow ───────────────────────────────────────────────────────────

def _log_partition_metrics(metrics: PartitionMetrics) -> None:
    """Registrar métricas de una partición en el run activo de MLflow."""
    p = metrics.partition
    mlflow.log_metrics({
        f"{p}_auc_roc":      metrics.auc_roc,
        f"{p}_ks":           metrics.ks_stat,
        f"{p}_ks_threshold": metrics.ks_threshold,
        f"{p}_precision_05": metrics.precision_05,
        f"{p}_recall_05":    metrics.recall_05,
        f"{p}_f1_05":        metrics.f1_05,
        f"{p}_precision_ks": metrics.precision_ks,
        f"{p}_recall_ks":    metrics.recall_ks,
        f"{p}_f1_ks":        metrics.f1_ks,
        f"{p}_auc_pr":       metrics.auc_pr,
        f"{p}_n_samples":    float(metrics.n_samples),
        f"{p}_n_positive":   float(metrics.n_positive),
        f"{p}_prevalence":   metrics.prevalence,
    })


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_training_pipeline(
    data_dir: str,
    mlflow_tracking_uri: str = "http://localhost:5000",
) -> None:
    """
    Pipeline completo: carga splits → entrena → evalúa en Train/Test/OOT → MLflow.

    Parameters
    ----------
    data_dir : str
        Directorio con train.parquet, test.parquet, oot.parquet, feature_cols.json.
    mlflow_tracking_uri : str
        URI del servidor de tracking MLflow.

    Notes
    -----
    S7: las particiones ya están definidas temporalmente por feature_engineering step.
    S8: imbalance 3.7:1 corregido vía scale_pos_weight.
    """
    data_path = Path(data_dir)

    df_train = pd.read_parquet(data_path / "train.parquet")
    df_test  = pd.read_parquet(data_path / "test.parquet")
    df_oot   = pd.read_parquet(data_path / "oot.parquet")

    with open(data_path / "feature_cols.json") as fh:
        feature_cols: list[str] = json.load(fh)

    logger.info(
        "Particiones cargadas | train=%d test=%d oot=%d | features=%d",
        len(df_train), len(df_test), len(df_oot), len(feature_cols),
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_debit_v1") as run:
        # Parámetros del experimento
        mlflow.log_params({
            "model_type":  "xgboost",
            "n_features":  len(feature_cols),
            "n_train":     len(df_train),
            "n_test":      len(df_test),
            "n_oot":       len(df_oot),
            "target_col":  TARGET_COL,
            "experiment":  EXPERIMENT_NAME,
            **{f"xgb_{k}": v for k, v in XGBOOST_PARAMS.items()
               if k not in ("early_stopping_rounds", "eval_metric")},
        })

        # Entrenamiento
        model = train_xgboost(df_train, feature_cols)

        mlflow.log_params({
            "best_iteration":        model.best_iteration,
            "scale_pos_weight_used": round(model.get_params()["scale_pos_weight"], 4),
        })

        # Evaluación en las 3 particiones
        all_metrics: list[PartitionMetrics] = []
        for name, df_part in [("train", df_train), ("test", df_test), ("oot", df_oot)]:
            m = evaluate_partition(model, df_part, feature_cols, name)
            _log_partition_metrics(m)
            all_metrics.append(m)

        # Feature importance — top 30
        importance = pd.Series(
            model.feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)

        top30 = importance.head(30)
        mlflow.log_param(
            "top_features",
            json.dumps({k: round(float(v), 5) for k, v in top30.items()}),
        )

        # Guardar feature importance como artefacto
        fi_path = data_path / "feature_importance.csv"
        importance.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ).to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # Guardar modelo
        model_path = data_path / "model_xgboost.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump({"model": model, "feature_cols": feature_cols}, fh)
        mlflow.log_artifact(str(model_path))
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # Resumen en log
        test_m = next(m for m in all_metrics if m.partition == "test")
        oot_m  = next(m for m in all_metrics if m.partition == "oot")
        logger.info(
            "=== RESULTADO FINAL ===\n"
            "  Test  — AUC=%.4f | KS=%.4f | F1@0.5=%.3f\n"
            "  OOT   — AUC=%.4f | KS=%.4f | F1@0.5=%.3f\n"
            "  run_id=%s",
            test_m.auc_roc, test_m.ks_stat, test_m.f1_05,
            oot_m.auc_roc,  oot_m.ks_stat,  oot_m.f1_05,
            run.info.run_id,
        )

    logger.info("Pipeline de entrenamiento completo. Modelo en %s", model_path)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Entrenar y evaluar clasificador de débitos recurrentes."
    )
    parser.add_argument("--data-dir",    required=True, help="Directorio con parquets de splits")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        mlflow_tracking_uri=args.mlflow_uri,
    )
