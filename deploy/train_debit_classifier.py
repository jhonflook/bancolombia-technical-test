#!/usr/bin/env python3
"""
Entrenamiento y comparación de clasificadores de débitos recurrentes.

Entrena múltiples clasificadores del MODEL_REGISTRY con optimización de
hiperparámetros vía Optuna, evalúa en Train / Test / OOT y registra el
mejor modelo en MLflow.

Uso:
    uv run python deploy/train_debit_classifier.py --data-dir data/artifacts
    uv run python deploy/train_debit_classifier.py --models xgboost random_forest
    uv run python deploy/train_debit_classifier.py --models xgboost --n-trials 50
    uv run python deploy/train_debit_classifier.py --models logistic_regression --n-trials 0

Modelos disponibles: xgboost, random_forest, gradient_boosting, logistic_regression
Default:             xgboost, random_forest, gradient_boosting
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import mlflow
import optuna
import pandas as pd

from src.statistical_models import DEFAULT_MODELS, MODEL_REGISTRY
from src.statistical_models.evaluation import compute_classification_metrics, evaluate_classifier_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENT_NAME = "debit_recurrence_classifier"
TARGET_COL = "var_rta"


def parse_args() -> argparse.Namespace:
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenar y comparar clasificadores de débitos recurrentes"
    )
    parser.add_argument(
        "--data-dir",
        default="data/artifacts",
        help="Directorio con train.parquet, test.parquet, oot.parquet y feature_cols.json",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="URI del servidor MLflow",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Trials Optuna por modelo (0 = solo parámetros por defecto)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            f"Modelos a entrenar. Disponibles: {', '.join(MODEL_REGISTRY.keys())}. "
            f"Default: {', '.join(DEFAULT_MODELS)}"
        ),
    )
    return parser.parse_args()


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Calcular cociente n_neg/n_pos para corrección de desbalance (S8)."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def optimize_hyperparams(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    n_trials: int,
    scale_pos_weight: float,
) -> dict:
    """Optimizar hiperparámetros de un clasificador con Optuna.

    Usa validación cruzada estratificada (5 folds) sobre el conjunto de
    entrenamiento. La métrica de optimización es AUC-ROC medio (S8).

    Parameters
    ----------
    model_name : str
        Clave en MODEL_REGISTRY.
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target binario de entrenamiento.
    feature_cols : list[str]
        Features seleccionados.
    n_trials : int
        Número de trials Optuna.
    scale_pos_weight : float
        Cociente n_neg/n_pos para corrección de desbalance.

    Returns
    -------
    dict
        Mejores hiperparámetros encontrados.
    """
    model_class = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = model_class().get_hyperparameter_space(trial)
        cv = evaluate_classifier_cv(
            model_class=model_class,
            X=X_train,
            y=y_train,
            selected_features=feature_cols,
            params=params,
            n_splits=5,
            scale_pos_weight=scale_pos_weight,
        )
        return cv.mean_auc

    study = optuna.create_study(direction="maximize", study_name=f"debit_{model_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "[%s] Optuna completado | best_cv_auc=%.4f | params=%s",
        model_name, study.best_value, study.best_params,
    )
    return study.best_params


def train_and_evaluate(
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    scale_pos_weight: float,
    data_path: Path,
) -> dict:
    """Entrenar modelo final y evaluar en Train / Test / OOT.

    Parameters
    ----------
    model_name : str
        Clave en MODEL_REGISTRY.
    df_train : pd.DataFrame
        Partición de entrenamiento (S7).
    df_test : pd.DataFrame
        Partición de test (S7).
    df_oot : pd.DataFrame
        Partición OOT — out-of-time (S7).
    feature_cols : list[str]
        Features seleccionados.
    best_params : dict
        Hiperparámetros optimizados (o vacío si n_trials=0).
    scale_pos_weight : float
        Cociente n_neg/n_pos para corrección de desbalance (S8).
    data_path : Path
        Directorio donde se guarda el artefacto pkl.

    Returns
    -------
    dict
        Diccionario con 'model', 'metrics', 'artifact_path', 'test_auc'.
    """
    model_class = MODEL_REGISTRY[model_name]
    clf = model_class()

    logger.info("[%s] Entrenando modelo final | params=%s", model_name, best_params)
    X_tr = df_train[feature_cols].fillna(0.0)
    y_tr = df_train[TARGET_COL].astype(int)
    clf.fit(X_tr, y_tr, selected_features=feature_cols,
            scale_pos_weight=scale_pos_weight, **best_params)

    all_metrics: dict[str, float] = {}
    for partition, df_part in [("train", df_train), ("test", df_test), ("oot", df_oot)]:
        X_part = df_part[feature_cols].fillna(0.0)
        y_part = df_part[TARGET_COL].astype(int)
        y_prob = clf.predict_proba(X_part)

        part_metrics = compute_classification_metrics(y_part.values, y_prob, partition=partition)
        all_metrics.update(part_metrics)

        logger.info(
            "[%s | %s] n=%d | AUC=%.4f | KS=%.4f | F1@0.5=%.3f | AUC-PR=%.4f",
            model_name, partition, len(df_part),
            part_metrics[f"{partition}_auc_roc"],
            part_metrics[f"{partition}_ks"],
            part_metrics[f"{partition}_f1_05"],
            part_metrics[f"{partition}_auc_pr"],
        )

    artifact_path = data_path / f"model_{model_name}.pkl"
    with open(artifact_path, "wb") as fh:
        pickle.dump({"model": clf, "feature_cols": feature_cols}, fh)

    return {
        "model":         clf,
        "metrics":       all_metrics,
        "artifact_path": artifact_path,
        "test_auc":      all_metrics["test_auc_roc"],
    }


def main() -> None:
    """Entrypoint principal: carga splits → optimiza → entrena → evalúa → MLflow."""
    args = parse_args()
    model_names = args.models or DEFAULT_MODELS
    data_path = Path(args.data_dir)

    # Cargar particiones temporales (S7)
    df_train = pd.read_parquet(data_path / "train.parquet")
    df_test  = pd.read_parquet(data_path / "test.parquet")
    df_oot   = pd.read_parquet(data_path / "oot.parquet")

    with open(data_path / "feature_cols.json") as fh:
        feature_cols: list[str] = json.load(fh)

    logger.info("=" * 70)
    logger.info("DEBIT RECURRENCE CLASSIFIER — ENTRENAMIENTO MULTI-MODELO")
    logger.info("=" * 70)
    logger.info("Modelos:   %s", model_names)
    logger.info("Features:  %d | train=%d | test=%d | oot=%d",
                len(feature_cols), len(df_train), len(df_test), len(df_oot))
    logger.info("n_trials:  %d", args.n_trials)

    scale_pos_weight = _compute_scale_pos_weight(df_train[TARGET_COL].astype(int))
    logger.info("scale_pos_weight=%.3f  (desbalance S8)", scale_pos_weight)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results: dict[str, dict] = {}

    with mlflow.start_run(run_name="debit_classifier_comparison") as parent_run:
        mlflow.log_params({
            "models":           ",".join(model_names),
            "n_trials":         args.n_trials,
            "n_features":       len(feature_cols),
            "n_train":          len(df_train),
            "n_test":           len(df_test),
            "n_oot":            len(df_oot),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "target_col":       TARGET_COL,
        })

        for model_name in model_names:
            logger.info("=" * 50)
            logger.info("MODELO: %s", model_name.upper())

            with mlflow.start_run(run_name=f"debit_{model_name}", nested=True):
                # Optimización de hiperparámetros con Optuna
                if args.n_trials > 0:
                    best_params = optimize_hyperparams(
                        model_name=model_name,
                        X_train=df_train[feature_cols].fillna(0.0),
                        y_train=df_train[TARGET_COL].astype(int),
                        feature_cols=feature_cols,
                        n_trials=args.n_trials,
                        scale_pos_weight=scale_pos_weight,
                    )
                else:
                    best_params = {}

                mlflow.log_params({"model_type": model_name, **best_params})

                # Entrenamiento final y evaluación en las 3 particiones
                result = train_and_evaluate(
                    model_name=model_name,
                    df_train=df_train,
                    df_test=df_test,
                    df_oot=df_oot,
                    feature_cols=feature_cols,
                    best_params=best_params,
                    scale_pos_weight=scale_pos_weight,
                    data_path=data_path,
                )
                results[model_name] = result

                mlflow.log_metrics(result["metrics"])
                mlflow.log_artifact(str(result["artifact_path"]))

                # Importancia de features (si el modelo la expone)
                importances = result["model"].get_feature_importances()
                if importances:
                    top30 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:30]
                    mlflow.log_param(
                        "top_features",
                        json.dumps({k: round(float(v), 5) for k, v in top30}),
                    )

        # Selección del mejor modelo por AUC-ROC en Test
        best_name = max(results, key=lambda m: results[m]["test_auc"])
        best_result = results[best_name]

        mlflow.log_params({"best_model": best_name})
        mlflow.log_metrics({
            "best_test_auc_roc": best_result["metrics"]["test_auc_roc"],
            "best_test_ks":      best_result["metrics"]["test_ks"],
            "best_oot_auc_roc":  best_result["metrics"]["oot_auc_roc"],
            "best_oot_ks":       best_result["metrics"]["oot_ks"],
        })

        logger.info("=" * 70)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 70)
        for name, res in results.items():
            marker = "  <-- MEJOR" if name == best_name else ""
            logger.info(
                "%-22s Test AUC=%.4f | Test KS=%.4f | OOT AUC=%.4f | OOT KS=%.4f%s",
                name,
                res["metrics"]["test_auc_roc"],
                res["metrics"]["test_ks"],
                res["metrics"]["oot_auc_roc"],
                res["metrics"]["oot_ks"],
                marker,
            )
        logger.info("MLflow run_id: %s", parent_run.info.run_id)
        logger.info("Ver resultados en: %s", args.mlflow_uri)


if __name__ == "__main__":
    main()
