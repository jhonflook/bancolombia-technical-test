#!/usr/bin/env python3
"""
Entrenamiento y comparación de clasificadores de débitos recurrentes.

Entrena múltiples clasificadores del MODEL_REGISTRY con optimización de
hiperparámetros vía Optuna. Estructura de runs MLflow:

  debit-models  (experimento MLflow)
  └── debit-models  (run padre — agrupa la comparación)
      ├── xgboost            ← run hijo anidado (nested=True)
      ├── random_forest      ← ídem
      └── gradient_boosting  ← ídem

Métricas con evolución temporal (step-based) por run hijo:
  cv_trial_auc   @ step=número de trial Optuna  → curva de optimización
  cv_fold_auc/ks/f1 @ step=0..4                 → estabilidad CV por fold
  evolution_auc_roc/ks/f1_05/auc_pr @ step=0(train)/1(test)/2(oot) → degradación

Figuras logueadas como artefactos (visibles en pestaña Artifacts de MLflow):
  figures/roc_curve.png           → curvas ROC train/test/oot superpuestas
  figures/pr_curve.png            → curvas Precision-Recall train/test/oot
  figures/feature_importance.png  → top-20 features por importancia
  figures/optuna_history.png      → evolución AUC por trial Optuna (si n_trials>0)

Métricas escalares por run hijo (para filtrado en MLflow):
  cv_auc, train_*, test_*, oot_*

Uso:
    uv run python deploy/train_debit_classifier.py --data-dir data/artifacts
    uv run python deploy/train_debit_classifier.py --models xgboost random_forest
    uv run python deploy/train_debit_classifier.py --models xgboost --n-trials 10
    uv run python deploy/train_debit_classifier.py --models logistic_regression --n-trials 0

Modelos disponibles: xgboost, random_forest, gradient_boosting, logistic_regression
Default:             xgboost, random_forest, gradient_boosting
"""

import argparse
import json
import logging
import pickle
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para entornos headless
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.statistical_models import DEFAULT_MODELS, MODEL_REGISTRY
from src.statistical_models.evaluation import (
    CVResults,
    compute_classification_metrics,
    evaluate_classifier_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENT_NAME = "debit-models"
TARGET_COL = "var_rta"

# Orden fijo de particiones para métricas de evolución (step=0,1,2)
PARTITION_STEPS = {"train": 0, "test": 1, "oot": 2}

# Métricas que se grafican como evolución Train→Test→OOT
EVOLUTION_METRICS = [
    "auc_roc", "ks", "f1_05", "auc_pr",
    "precision_05", "recall_05", "precision_ks", "recall_ks", "f1_ks",
]

# Colores por partición para figuras
_PARTITION_COLORS = {
    "train": "royalblue",
    "test":  "darkorange",
    "oot":   "forestgreen",
}


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
        help="URI del servidor MLflow (default: http://localhost:5000; "
             "usar http://mlflow:5000 dentro de la red Docker)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Trials Optuna por modelo (0 = solo parámetros por defecto, sí corre 1 CV)",
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
    parser.add_argument(
        "--split-strategy",
        default="temporal",
        choices=["temporal", "random"],
        help=(
            "Estrategia usada para generar los parquets de entrada. "
            "Solo se loguea en MLflow; no altera la carga de datos. "
            "Debe coincidir con el --split-strategy usado en feature_engineering. "
            "Default: 'temporal'"
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
) -> tuple[dict, float, list[tuple[int, float]], CVResults | None]:
    """Optimizar hiperparámetros con Optuna y capturar historial de trials.

    Parameters
    ----------
    model_name : str
    X_train : pd.DataFrame
    y_train : pd.Series
    feature_cols : list[str]
    n_trials : int
    scale_pos_weight : float

    Returns
    -------
    tuple[dict, float, list[tuple[int, float]], CVResults | None]
        (mejores params, mejor AUC CV, historial [(trial_num, auc)], CVResults del mejor trial)
    """
    model_class = MODEL_REGISTRY[model_name]
    trial_history: list[tuple[int, float]] = []
    best_cv_results: CVResults | None = None
    best_auc_so_far = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_cv_results, best_auc_so_far
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
        trial_history.append((trial.number, cv.mean_auc))
        if cv.mean_auc > best_auc_so_far:
            best_auc_so_far = cv.mean_auc
            best_cv_results = cv
        return cv.mean_auc

    study = optuna.create_study(direction="maximize", study_name=f"debit_{model_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "[%s] Optuna completado | best_cv_auc=%.4f | params=%s",
        model_name, study.best_value, study.best_params,
    )
    return study.best_params, study.best_value, trial_history, best_cv_results


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
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    df_oot : pd.DataFrame
    feature_cols : list[str]
    best_params : dict
    scale_pos_weight : float
    data_path : Path

    Returns
    -------
    dict
        Claves: model, metrics, artifact_path, test_auc, importances
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
        "importances":   clf.get_feature_importances(),
    }


def _log_figures(
    clf,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    importances: dict | None,
    trial_history: list[tuple[int, float]],
) -> None:
    """Loguear figuras matplotlib como artefactos en el run MLflow activo.

    Genera y sube:
    - ROC curves superpuestas (train/test/oot)
    - Precision-Recall curves superpuestas (train/test/oot)
    - Feature importance bar chart (top 20)
    - Optuna optimization history (si trial_history no está vacío)

    Cada figura se envuelve en try/except para no interrumpir el run.
    """
    partitions = [
        ("train", df_train),
        ("test",  df_test),
        ("oot",   df_oot),
    ]

    # ── ROC curves ───────────────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        for pname, df_p in partitions:
            y_true = df_p[TARGET_COL].astype(int).values
            y_prob = clf.predict_proba(df_p[feature_cols].fillna(0.0))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{pname} (AUC={auc_val:.3f})",
                    color=_PARTITION_COLORS[pname], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/roc_curve.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear ROC curve: %s", model_name, exc)

    # ── Precision-Recall curves ──────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        for pname, df_p in partitions:
            y_true = df_p[TARGET_COL].astype(int).values
            y_prob = clf.predict_proba(df_p[feature_cols].fillna(0.0))
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = float(sklearn_auc(rec, prec))
            ax.plot(rec, prec, label=f"{pname} (AUC-PR={auc_pr:.3f})",
                    color=_PARTITION_COLORS[pname], linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {model_name}")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/pr_curve.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear PR curve: %s", model_name, exc)

    # ── Feature importance (top 20) ──────────────────────────────────────────
    if importances:
        try:
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
            names = [x[0] for x in sorted_imp]
            vals  = [x[1] for x in sorted_imp]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(names[::-1], vals[::-1], color="steelblue", edgecolor="white")
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance (Top 20) — {model_name}")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/feature_importance.png")
            plt.close(fig)
        except Exception as exc:
            logger.warning("[%s] No se pudo loguear feature importance: %s", model_name, exc)

    # ── Optuna optimization history ──────────────────────────────────────────
    if trial_history:
        try:
            trial_nums = [t[0] for t in trial_history]
            trial_aucs = [t[1] for t in trial_history]
            best_so_far = [max(trial_aucs[:i + 1]) for i in range(len(trial_aucs))]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(trial_nums, trial_aucs, "o-", alpha=0.55,
                    label="CV AUC (trial)", color="steelblue")
            ax.plot(trial_nums, best_so_far, "r--",
                    label="Mejor acumulado", linewidth=2)
            ax.set_xlabel("Trial")
            ax.set_ylabel("CV AUC (StratifiedKFold 5)")
            ax.set_title(f"Optuna Optimization History — {model_name}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/optuna_history.png")
            plt.close(fig)
        except Exception as exc:
            logger.warning("[%s] No se pudo loguear Optuna history: %s", model_name, exc)


def _log_model_run(
    model_name: str,
    best_params: dict,
    cv_auc: float,
    result: dict,
    scale_pos_weight: float,
    n_features: int,
    n_trials: int,
    trial_history: list[tuple[int, float]],
    cv_fold_results: pd.DataFrame | None,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    feature_cols: list[str],
) -> str:
    """Registrar run hijo anidado en MLflow para un modelo.

    Loguea dentro de un run activo padre usando nested=True.
    Incluye evolución por trial Optuna, por fold CV, Train→Test→OOT y figuras.

    Parameters
    ----------
    model_name : str
    best_params : dict
    cv_auc : float
    result : dict
    scale_pos_weight : float
    n_features : int
    n_trials : int
    trial_history : list[tuple[int, float]]
        [(trial_number, cv_auc), ...] — historial de optimización Optuna.
    cv_fold_results : pd.DataFrame | None
        fold_results del mejor trial: columnas fold, auc_roc, ks_stat, f1_05.
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    df_oot : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    str
        run_id del run hijo creado.
    """
    with mlflow.start_run(run_name=model_name, nested=True) as run:

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": model_name,
            "framework":  "sklearn-compatible",
        })

        # ── Parámetros ────────────────────────────────────────────────────────
        mlflow.log_params({
            "model_type":       model_name,
            "n_features":       n_features,
            "n_trials":         n_trials,
            "scale_pos_weight": round(scale_pos_weight, 4),
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in best_params.items()},
        })

        # ── CV AUC escalar ────────────────────────────────────────────────────
        mlflow.log_metric("cv_auc", round(cv_auc, 4))

        # ── Evolución Optuna: AUC por trial (step = número de trial) ─────────
        for trial_num, trial_auc in trial_history:
            mlflow.log_metric("cv_trial_auc", round(trial_auc, 4), step=trial_num)

        # ── Evolución CV: métricas por fold (step = fold 0..4) ───────────────
        if cv_fold_results is not None:
            for _, row in cv_fold_results.iterrows():
                step = int(row["fold"])
                mlflow.log_metric("cv_fold_auc", round(float(row["auc_roc"]), 4), step=step)
                mlflow.log_metric("cv_fold_ks",  round(float(row["ks_stat"]), 4), step=step)
                mlflow.log_metric("cv_fold_f1",  round(float(row["f1_05"]),   4), step=step)

        # ── Métricas escalares completas: train / test / OOT ─────────────────
        mlflow.log_metrics(result["metrics"])

        # ── Evolución particiones Train→Test→OOT (step=0,1,2) ────────────────
        # Permite graficar curvas de generalización en la UI de MLflow.
        for partition, step in PARTITION_STEPS.items():
            for metric in EVOLUTION_METRICS:
                key = f"{partition}_{metric}"
                if key in result["metrics"]:
                    mlflow.log_metric(
                        f"evolution_{metric}",
                        result["metrics"][key],
                        step=step,
                    )

        # ── Figuras matplotlib como artefactos ───────────────────────────────
        _log_figures(
            clf=result["model"],
            df_train=df_train,
            df_test=df_test,
            df_oot=df_oot,
            feature_cols=feature_cols,
            model_name=model_name,
            importances=result["importances"],
            trial_history=trial_history,
        )

        # ── Artefacto: modelo pkl ─────────────────────────────────────────────
        try:
            mlflow.log_artifact(str(result["artifact_path"]))
        except Exception as exc:
            logger.warning("[%s] No se pudo subir el artefacto pkl: %s", model_name, exc)

        # ── Artefacto: feature importances CSV ───────────────────────────────
        importances = result["importances"]
        if importances:
            rows = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            imp_df = pd.DataFrame(rows, columns=["feature", "importance"])
            with tempfile.NamedTemporaryFile(
                suffix=".csv", prefix=f"feature_importance_{model_name}_",
                delete=False, mode="w",
            ) as tmp:
                imp_df.to_csv(tmp.name, index=False)
                try:
                    mlflow.log_artifact(tmp.name, artifact_path="feature_importance")
                except Exception as exc:
                    logger.warning(
                        "[%s] No se pudo subir feature_importance.csv: %s", model_name, exc
                    )

        return run.info.run_id


def main() -> None:
    """Entrypoint: carga splits → optimiza → entrena → evalúa → MLflow."""
    args = parse_args()
    model_names = args.models or DEFAULT_MODELS
    data_path   = Path(args.data_dir)

    df_train = pd.read_parquet(data_path / "train.parquet")
    df_test  = pd.read_parquet(data_path / "test.parquet")
    df_oot   = pd.read_parquet(data_path / "oot.parquet")

    with open(data_path / "feature_cols.json") as fh:
        feature_cols: list[str] = json.load(fh)

    logger.info("=" * 70)
    logger.info("DEBIT RECURRENCE CLASSIFIER — ENTRENAMIENTO MULTI-MODELO")
    logger.info("=" * 70)
    logger.info("Modelos:        %s", model_names)
    logger.info("Split strategy: %s", args.split_strategy)
    logger.info("Features:       %d | train=%d | test=%d | oot=%d",
                len(feature_cols), len(df_train), len(df_test), len(df_oot))
    logger.info("n_trials:       %d", args.n_trials)

    scale_pos_weight = _compute_scale_pos_weight(df_train[TARGET_COL].astype(int))
    logger.info("scale_pos_weight=%.3f  (desbalance S8)", scale_pos_weight)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results:  dict[str, dict]  = {}
    cv_aucs:  dict[str, float] = {}
    run_ids:  dict[str, str]   = {}

    # ── Run padre: agrupa todos los modelos ──────────────────────────────────
    with mlflow.start_run(run_name="debit-models") as parent_run:
        mlflow.log_params({
            "models_compared":  ",".join(model_names),
            "n_trials":         args.n_trials,
            "n_features":       len(feature_cols),
            "train_size":       len(df_train),
            "test_size":        len(df_test),
            "oot_size":         len(df_oot),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "split_strategy":   args.split_strategy,
        })

        # ── Un run hijo por modelo (nested=True) ─────────────────────────────
        for model_name in model_names:
            logger.info("=" * 50)
            logger.info("MODELO: %s", model_name.upper())

            try:
                if args.n_trials > 0:
                    best_params, cv_auc, trial_history, best_cv = optimize_hyperparams(
                        model_name=model_name,
                        X_train=df_train[feature_cols].fillna(0.0),
                        y_train=df_train[TARGET_COL].astype(int),
                        feature_cols=feature_cols,
                        n_trials=args.n_trials,
                        scale_pos_weight=scale_pos_weight,
                    )
                    cv_fold_results = best_cv.fold_results if best_cv is not None else None
                else:
                    # Sin Optuna: parámetros por defecto + 1 CV para fold metrics
                    best_params = {}
                    trial_history = []
                    cv_single = evaluate_classifier_cv(
                        model_class=MODEL_REGISTRY[model_name],
                        X=df_train[feature_cols].fillna(0.0),
                        y=df_train[TARGET_COL].astype(int),
                        selected_features=feature_cols,
                        params={},
                        n_splits=5,
                        scale_pos_weight=scale_pos_weight,
                    )
                    cv_auc = cv_single.mean_auc
                    cv_fold_results = cv_single.fold_results

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
                results[model_name]  = result
                cv_aucs[model_name]  = cv_auc
                run_ids[model_name]  = _log_model_run(
                    model_name=model_name,
                    best_params=best_params,
                    cv_auc=cv_auc,
                    result=result,
                    scale_pos_weight=scale_pos_weight,
                    n_features=len(feature_cols),
                    n_trials=args.n_trials,
                    trial_history=trial_history,
                    cv_fold_results=cv_fold_results,
                    df_train=df_train,
                    df_test=df_test,
                    df_oot=df_oot,
                    feature_cols=feature_cols,
                )

            except Exception as exc:
                logger.error("[%s] Falló: %s", model_name, exc, exc_info=True)
                with mlflow.start_run(run_name=model_name, nested=True):
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", str(exc)[:250])

        if not results:
            logger.error("Ningún modelo completó el entrenamiento.")
            mlflow.set_tag("status", "ALL_FAILED")
            return

        # ── Comparación: métricas resumen en el run padre ─────────────────────
        best_name   = max(results, key=lambda m: results[m]["test_auc"])
        best_result = results[best_name]

        mlflow.log_param("best_model", best_name)

        comparison_metrics: dict[str, float] = {}
        for name, res in results.items():
            prefix = name.replace("_", "")
            comparison_metrics.update({
                f"{prefix}_cv_auc":    round(cv_aucs[name], 4),
                f"{prefix}_train_auc": round(res["metrics"]["train_auc_roc"], 4),
                f"{prefix}_train_ks":  round(res["metrics"]["train_ks"], 4),
                f"{prefix}_test_auc":  round(res["metrics"]["test_auc_roc"], 4),
                f"{prefix}_test_ks":   round(res["metrics"]["test_ks"], 4),
                f"{prefix}_oot_auc":   round(res["metrics"]["oot_auc_roc"], 4),
                f"{prefix}_oot_ks":    round(res["metrics"]["oot_ks"], 4),
            })
        mlflow.log_metrics(comparison_metrics)

        mlflow.log_metrics({
            "best_cv_auc":    round(cv_aucs[best_name], 4),
            "best_train_auc": round(best_result["metrics"]["train_auc_roc"], 4),
            "best_train_ks":  round(best_result["metrics"]["train_ks"], 4),
            "best_test_auc":  round(best_result["metrics"]["test_auc_roc"], 4),
            "best_test_ks":   round(best_result["metrics"]["test_ks"], 4),
            "best_oot_auc":   round(best_result["metrics"]["oot_auc_roc"], 4),
            "best_oot_ks":    round(best_result["metrics"]["oot_ks"], 4),
        })

    # ── Resumen en log ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    logger.info("%-22s %8s %8s %8s %8s %8s %8s",
                "Modelo", "CV AUC", "Tr AUC", "Tr KS", "Te AUC", "Te KS", "OOT AUC")
    for name, res in results.items():
        marker = "  <-- MEJOR" if name == best_name else ""
        logger.info(
            "%-22s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f%s",
            name,
            cv_aucs[name],
            res["metrics"]["train_auc_roc"],
            res["metrics"]["train_ks"],
            res["metrics"]["test_auc_roc"],
            res["metrics"]["test_ks"],
            res["metrics"]["oot_auc_roc"],
            marker,
        )
    logger.info(
        "MLflow: %s  |  Experimento: %s  |  Parent run: %s",
        args.mlflow_uri, EXPERIMENT_NAME, parent_run.info.run_id,
    )


if __name__ == "__main__":
    main()
