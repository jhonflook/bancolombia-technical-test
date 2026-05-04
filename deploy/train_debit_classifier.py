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
  figures/roc_curve.png               → curvas ROC train/test/oot superpuestas
  figures/pr_curve.png                → curvas Precision-Recall train/test/oot
  figures/feature_importance.png      → top-20 features por importancia nativa
  figures/optuna_history.png          → evolución AUC por trial Optuna (si n_trials>0)
  figures/score_distribution.png      → histograma de scores por clase (train/test/oot)
  figures/confusion_matrix.png        → matriz de confusión en test al umbral KS
  figures/ks_lift_chart.png           → curva de ganancias + lift por decil (test)
  figures/training_curve.png          → AUC por ronda (XGBoost) o pérdida por iter (GBM)
  figures/calibration_curve.png       → diagrama de calibración con Brier score (test)
  figures/shap_summary.png            → SHAP beeswarm top-20 (todos los modelos)
  figures/shap_bar.png                → SHAP mean|SHAP| bar chart top-20
  figures/shap_dependence_<feat>.png  → SHAP dependence plot top-3 features

SHAP: TreeExplainer (xgboost/gradient_boosting/random_forest), LinearExplainer (logistic_regression)

Métricas escalares por run hijo (para filtrado en MLflow):
  cv_auc, train_*, test_*, oot_*
  {partition}_brier, {partition}_log_loss    → calibración por partición
  training_time_sec, model_size_kb           → recursos de entrenamiento
  n_rounds_used                              → rondas XGBoost tras early stopping

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
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para entornos headless
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc as sklearn_auc,
    brier_score_loss,
    confusion_matrix,
    log_loss as sklearn_log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.dataset.feature_config import DEFAULT_FEATURE_CONFIG, OPTION_B_EXCLUDE_GROUPS
from src.statistical_models import DEFAULT_MODELS, MODEL_REGISTRY
from src.statistical_models.evaluation import (
    CVResults,
    compute_classification_metrics,
    compute_ks,
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
    parser.add_argument(
        "--feature-set",
        default="A",
        choices=["A", "B", "both"],
        help=(
            "Conjunto de features a usar. "
            "A: todas las features seleccionadas (incluye débito). "
            "B: excluye grupos deterministas por H1 (pagos, excedentes, canales). "
            "both: entrena A y B en runs MLflow separados. "
            "Default: 'A'"
        ),
    )
    return parser.parse_args()


def _filter_option_b(feature_cols: list[str]) -> list[str]:
    """Filtra los grupos deterministas por H1 para Opción B.

    Elimina pagos, excedentes, canales y sus derivadas — todas con valor 0
    para clase-0 (sin actividad de pago). Retiene gestiones + moras.
    """
    exclude = set(
        feat
        for g in OPTION_B_EXCLUDE_GROUPS
        for feat in DEFAULT_FEATURE_CONFIG.get_group(g)
    )
    kept = [f for f in feature_cols if f not in exclude]
    logger.info(
        "Opción B: %d features excluidas → %d features retenidas (gestiones + moras)",
        len(feature_cols) - len(kept), len(kept),
    )
    return kept


def _build_run_label(split_strategy: str, feature_set: str) -> str:
    """Etiqueta legible para títulos de figuras y nombres de runs MLflow."""
    split_label = "aleatorio" if split_strategy == "random" else "temporal"
    return f"split {split_label} | Opción {feature_set}"


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
    pkl_suffix: str = "",
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
    t0 = time.time()
    clf.fit(X_tr, y_tr, selected_features=feature_cols,
            scale_pos_weight=scale_pos_weight, expose_history=True, **best_params)
    training_time_sec = round(time.time() - t0, 2)

    all_metrics: dict[str, float] = {}
    for partition, df_part in [("train", df_train), ("test", df_test), ("oot", df_oot)]:
        X_part = df_part[feature_cols].fillna(0.0)
        y_part = df_part[TARGET_COL].astype(int)
        y_prob = clf.predict_proba(X_part)

        part_metrics = compute_classification_metrics(y_part.values, y_prob, partition=partition)
        all_metrics.update(part_metrics)
        all_metrics[f"{partition}_brier"]    = round(float(brier_score_loss(y_part.values, y_prob)), 4)
        all_metrics[f"{partition}_log_loss"] = round(float(sklearn_log_loss(y_part.values, y_prob)), 4)

        logger.info(
            "[%s | %s] n=%d | AUC=%.4f | KS=%.4f | F1@0.5=%.3f | AUC-PR=%.4f | Brier=%.4f",
            model_name, partition, len(df_part),
            part_metrics[f"{partition}_auc_roc"],
            part_metrics[f"{partition}_ks"],
            part_metrics[f"{partition}_f1_05"],
            part_metrics[f"{partition}_auc_pr"],
            all_metrics[f"{partition}_brier"],
        )

    artifact_path = data_path / f"model_{model_name}{pkl_suffix}.pkl"
    with open(artifact_path, "wb") as fh:
        pickle.dump({"model": clf, "feature_cols": feature_cols}, fh)
    model_size_kb = round(artifact_path.stat().st_size / 1024, 1)

    training_history = clf.get_training_history()
    if training_history and "val_auc" in training_history:
        all_metrics["n_rounds_used"] = len(training_history["val_auc"])

    return {
        "model":             clf,
        "metrics":           all_metrics,
        "artifact_path":     artifact_path,
        "test_auc":          all_metrics["test_auc_roc"],
        "importances":       clf.get_feature_importances(),
        "training_history":  training_history,
        "training_time_sec": training_time_sec,
        "model_size_kb":     model_size_kb,
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
    run_label: str = "",
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
        ax.set_title(f"ROC Curve — {model_name}\n[{run_label}]")
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
        ax.set_title(f"Precision-Recall Curve — {model_name}\n[{run_label}]")
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
            ax.set_title(f"Feature Importance (Top 20) — {model_name}\n[{run_label}]")
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
            ax.set_title(f"Optuna Optimization History — {model_name}\n[{run_label}]")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/optuna_history.png")
            plt.close(fig)
        except Exception as exc:
            logger.warning("[%s] No se pudo loguear Optuna history: %s", model_name, exc)

    # ── Score distribution (histograma de scores por clase) ──────────────────
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (pname, df_p) in zip(axes, partitions):
            y_true = df_p[TARGET_COL].astype(int).values
            y_prob = clf.predict_proba(df_p[feature_cols].fillna(0.0))
            ax.hist(y_prob[y_true == 0], bins=40, alpha=0.65,
                    label="Clase 0 (sin débito)", color="crimson", density=True)
            ax.hist(y_prob[y_true == 1], bins=40, alpha=0.65,
                    label="Clase 1 (débito exc.)", color="steelblue", density=True)
            ax.set_title(f"{pname.upper()}")
            ax.set_xlabel("P(débito recurrente)")
            ax.set_ylabel("Densidad")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Score Distribution — {model_name}  [{run_label}]",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/score_distribution.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear score distribution: %s", model_name, exc)

    # ── Confusion matrix en test al umbral KS ───────────────────────────────
    try:
        y_true_t = df_test[TARGET_COL].astype(int).values
        y_prob_t = clf.predict_proba(df_test[feature_cols].fillna(0.0))
        _, ks_thr = compute_ks(y_true_t, y_prob_t)
        y_pred_ks = (y_prob_t >= ks_thr).astype(int)
        cm = confusion_matrix(y_true_t, y_pred_ks)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Clase 0", "Clase 1"])
        disp.plot(ax=ax, colormap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix — Test (thr KS={ks_thr:.3f})\n{model_name}  [{run_label}]")
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/confusion_matrix.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear confusion matrix: %s", model_name, exc)

    # ── KS Lift chart (curva de ganancias acumuladas) ────────────────────────
    try:
        y_true_t = df_test[TARGET_COL].astype(int).values
        y_prob_t = clf.predict_proba(df_test[feature_cols].fillna(0.0))
        order = np.argsort(-y_prob_t)
        n_total   = len(y_true_t)
        n_pos     = y_true_t.sum()
        pct_pop   = np.arange(1, n_total + 1) / n_total
        cum_gains = y_true_t[order].cumsum() / n_pos
        baseline  = pct_pop
        lift      = cum_gains / np.where(baseline > 0, baseline, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(pct_pop * 100, cum_gains * 100, color="steelblue", linewidth=2,
                     label="Modelo")
        axes[0].plot([0, 100], [0, 100], "k--", alpha=0.4, label="Aleatorio")
        axes[0].set_xlabel("% Población contactada")
        axes[0].set_ylabel("% Clase 1 capturada")
        axes[0].set_title("Curva de Ganancias (Test)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        decile_pct = np.linspace(10, 100, 10)
        decile_lift = [lift[int(p / 100 * n_total) - 1] for p in decile_pct]
        axes[1].bar(decile_pct, decile_lift, width=8, color="steelblue",
                    edgecolor="white", alpha=0.85)
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5,
                        label="Baseline (lift=1)")
        axes[1].set_xlabel("Decil (% población)")
        axes[1].set_ylabel("Lift")
        axes[1].set_title("Lift por Decil (Test)")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"KS Lift Chart — {model_name}  [{run_label}]",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/ks_lift_chart.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear KS lift chart: %s", model_name, exc)

    # ── SHAP plots — todos los modelos ──────────────────────────────────────
    # Tree models: TreeExplainer | logistic_regression: LinearExplainer (scaled)
    try:
        import shap
        inner_model = getattr(clf, "model", None)
        if inner_model is not None:
            X_shap_raw = (
                df_test[feature_cols]
                .fillna(0.0)
                .sample(min(800, len(df_test)), random_state=42)
            )

            if model_name in ("xgboost", "gradient_boosting", "random_forest"):
                explainer   = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X_shap_raw, check_additivity=False)
                X_shap_disp = X_shap_raw
            elif model_name == "logistic_regression":
                scaler = getattr(clf, "scaler", None)
                X_scaled = pd.DataFrame(
                    scaler.transform(X_shap_raw) if scaler is not None else X_shap_raw.values,
                    columns=feature_cols,
                    index=X_shap_raw.index,
                )
                explainer   = shap.LinearExplainer(inner_model, X_scaled)
                shap_values = explainer.shap_values(X_scaled)
                X_shap_disp = X_scaled
            else:
                raise ValueError(f"Explainer no definido para: {model_name}")

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Beeswarm (summary_plot tipo dot)
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap_disp, show=False,
                              max_display=20, plot_type="dot")
            plt.title(f"SHAP Summary (beeswarm) — {model_name}\n[{run_label}]",
                      fontsize=12, fontweight="bold")
            plt.tight_layout()
            mlflow.log_figure(fig, "figures/shap_summary.png")
            plt.close(fig)

            # Bar (importancia media absoluta SHAP)
            fig = plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_shap_disp, show=False,
                              max_display=20, plot_type="bar")
            plt.title(f"SHAP Feature Importance (mean |SHAP|) — {model_name}\n[{run_label}]", fontsize=12)
            plt.tight_layout()
            mlflow.log_figure(fig, "figures/shap_bar.png")
            plt.close(fig)

            # Dependence plots — top 3 features por mean |SHAP|
            mean_abs    = np.abs(shap_values).mean(axis=0)
            top3_idx    = np.argsort(mean_abs)[-3:][::-1]
            top3_feats  = [feature_cols[i] for i in top3_idx]
            for feat in top3_feats:
                try:
                    safe = feat.replace("/", "_").replace(" ", "_")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.dependence_plot(
                        feat, shap_values, X_shap_disp,
                        ax=ax, show=False, interaction_index=None,
                        dot_size=12, alpha=0.6,
                    )
                    ax.set_title(f"SHAP Dependence — {feat}\n{model_name}  [{run_label}]", fontsize=11)
                    fig.tight_layout()
                    mlflow.log_figure(fig, f"figures/shap_dependence_{safe}.png")
                    plt.close(fig)
                except Exception as dep_exc:
                    logger.debug("[%s] SHAP dependence '%s': %s", model_name, feat, dep_exc)

            logger.info("[%s] SHAP plots logueados (%d features).", model_name, len(feature_cols))
    except Exception as exc:
        logger.warning("[%s] SHAP plots fallaron: %s", model_name, exc)


def _log_training_curve_figure(
    model_name: str, training_history: dict | None, run_label: str = ""
) -> None:
    """Loguear figura de evolución métrica/pérdida por iteración en el run MLflow activo."""
    if not training_history:
        return
    try:
        if "val_auc" in training_history:
            val_auc  = training_history["val_auc"]
            rounds   = list(range(len(val_auc)))
            best_r   = int(np.argmax(val_auc))
            fig, ax  = plt.subplots(figsize=(9, 5))
            ax.plot(rounds, val_auc, color="darkorange", linewidth=2,
                    label="AUC validación interna")
            ax.fill_between(rounds, val_auc, alpha=0.12, color="darkorange")
            ax.axvline(best_r, color="crimson", linestyle="--", alpha=0.75,
                       label=f"Mejor ronda={best_r}  AUC={val_auc[best_r]:.4f}")
            ax.set_xlabel("Ronda de boosting")
            ax.set_ylabel("AUC (validación interna 15%)")
            ax.set_title(f"Curva de Entrenamiento — {model_name}\n[{run_label}]", fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/training_curve.png")
            plt.close(fig)

        elif "train_loss" in training_history:
            train_loss = training_history["train_loss"]
            val_loss   = training_history.get("val_loss", [])
            iters      = list(range(len(train_loss)))
            fig, ax    = plt.subplots(figsize=(9, 5))
            ax.plot(iters, train_loss, color="royalblue", linewidth=2, label="Train loss")
            if val_loss:
                v_iters = list(range(len(val_loss)))
                ax.plot(v_iters, val_loss, color="darkorange", linewidth=2,
                        label="Val loss (early stopping)")
                best_i = int(np.argmin(val_loss))
                ax.axvline(best_i, color="crimson", linestyle="--", alpha=0.75,
                           label=f"Mejor iter={best_i}  val_loss={val_loss[best_i]:.4f}")
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Log-loss")
            ax.set_title(f"Curva de Pérdida — {model_name}\n[{run_label}]", fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "figures/training_curve.png")
            plt.close(fig)

    except Exception as exc:
        logger.warning("[%s] No se pudo loguear training curve: %s", model_name, exc)


def _log_calibration_figure(
    clf,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    run_label: str = "",
) -> None:
    """Loguear diagrama de calibración (reliability diagram) sobre la partición test."""
    try:
        from sklearn.calibration import calibration_curve
        y_true = df_test[TARGET_COL].astype(int).values
        y_prob = clf.predict_proba(df_test[feature_cols].fillna(0.0))
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        brier = float(brier_score_loss(y_true, y_prob))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(mean_pred, frac_pos, "s-", color="steelblue", linewidth=2, label="Modelo")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectamente calibrado")
        ax.set_xlabel("Probabilidad predicha (media del bin)")
        ax.set_ylabel("Fracción de positivos observados")
        ax.set_title(
            f"Diagrama de Calibración (Test) — {model_name}\nBrier={brier:.4f}  [{run_label}]",
            fontweight="bold",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "figures/calibration_curve.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("[%s] No se pudo loguear calibration curve: %s", model_name, exc)


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
    split_strategy: str = "temporal",
    feature_set: str = "A",
    run_label: str = "",
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
            "model_type":          model_name,
            "framework":           "sklearn-compatible",
            "split_strategy":      split_strategy,
            "feature_set":         feature_set,
            "run_label":           run_label,
            "target_rate_train":   f"{df_train[TARGET_COL].mean():.3f}",
            "target_rate_test":    f"{df_test[TARGET_COL].mean():.3f}",
            "target_rate_oot":     f"{df_oot[TARGET_COL].mean():.3f}",
            "n_train":             str(len(df_train)),
            "n_test":              str(len(df_test)),
            "n_oot":               str(len(df_oot)),
        })

        # ── Parámetros ────────────────────────────────────────────────────────
        mlflow.log_params({
            "model_type":       model_name,
            "n_features":       n_features,
            "n_trials":         n_trials,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "split_strategy":   split_strategy,
            "feature_set":      feature_set,
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

        # ── Métricas adicionales: tiempo, tamaño, rondas usadas ─────────────
        for extra_key in ("training_time_sec", "model_size_kb"):
            val = result.get(extra_key)
            if val is not None:
                mlflow.log_metric(extra_key, val)

        # ── Curva de entrenamiento por iteración/ronda (step-based) ─────────
        training_history = result.get("training_history")
        if training_history:
            if "val_auc" in training_history:
                mlflow.log_metric("n_rounds_used", len(training_history["val_auc"]))
                for step, val in enumerate(training_history["val_auc"]):
                    mlflow.log_metric("xgb_val_auc_per_round", round(val, 4), step=step)
            if "train_loss" in training_history:
                for step, tl in enumerate(training_history["train_loss"]):
                    mlflow.log_metric("gbm_train_loss_per_iter", round(tl, 5), step=step)
                for step, vl in enumerate(training_history.get("val_loss", [])):
                    mlflow.log_metric("gbm_val_loss_per_iter", round(vl, 5), step=step)

        # ── Figura curva de entrenamiento ────────────────────────────────────
        _log_training_curve_figure(model_name, training_history, run_label=run_label)

        # ── Diagrama de calibración ──────────────────────────────────────────
        _log_calibration_figure(result["model"], df_test, feature_cols, model_name,
                                run_label=run_label)

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
            run_label=run_label,
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


def _run_feature_set(
    model_names: list[str],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    feature_cols: list[str],
    feature_set: str,
    split_strategy: str,
    n_trials: int,
    data_path: Path,
    pkl_suffix: str,
    scale_pos_weight: float,
) -> tuple[str, dict, dict]:
    """Ejecuta el loop completo de entrenamiento para un conjunto de features.

    Crea un run padre MLflow con nombre distintivo y runs hijos por modelo.

    Returns
    -------
    tuple[str, dict, dict]
        (parent_run_id, results_por_modelo, cv_aucs_por_modelo)
    """
    run_label   = _build_run_label(split_strategy, feature_set)
    parent_name = f"debit-models-{split_strategy}-opt{feature_set}"

    results: dict[str, dict]  = {}
    cv_aucs: dict[str, float] = {}

    with mlflow.start_run(run_name=parent_name) as parent_run:
        mlflow.log_params({
            "models_compared":  ",".join(model_names),
            "n_trials":         n_trials,
            "n_features":       len(feature_cols),
            "train_size":       len(df_train),
            "test_size":        len(df_test),
            "oot_size":         len(df_oot),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "split_strategy":   split_strategy,
            "feature_set":      feature_set,
        })
        mlflow.set_tags({
            "split_strategy":      split_strategy,
            "feature_set":         feature_set,
            "run_label":           run_label,
            "target_rate_train":   f"{df_train[TARGET_COL].mean():.3f}",
            "target_rate_test":    f"{df_test[TARGET_COL].mean():.3f}",
            "target_rate_oot":     f"{df_oot[TARGET_COL].mean():.3f}",
            "n_features_selected": str(len(feature_cols)),
        })

        for model_name in model_names:
            logger.info("=" * 50)
            logger.info("MODELO: %s  [%s]", model_name.upper(), run_label)

            try:
                if n_trials > 0:
                    best_params, cv_auc, trial_history, best_cv = optimize_hyperparams(
                        model_name=model_name,
                        X_train=df_train[feature_cols].fillna(0.0),
                        y_train=df_train[TARGET_COL].astype(int),
                        feature_cols=feature_cols,
                        n_trials=n_trials,
                        scale_pos_weight=scale_pos_weight,
                    )
                    cv_fold_results = best_cv.fold_results if best_cv is not None else None
                else:
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
                    pkl_suffix=pkl_suffix,
                )
                results[model_name] = result
                cv_aucs[model_name] = cv_auc
                _log_model_run(
                    model_name=model_name,
                    best_params=best_params,
                    cv_auc=cv_auc,
                    result=result,
                    scale_pos_weight=scale_pos_weight,
                    n_features=len(feature_cols),
                    n_trials=n_trials,
                    trial_history=trial_history,
                    cv_fold_results=cv_fold_results,
                    df_train=df_train,
                    df_test=df_test,
                    df_oot=df_oot,
                    feature_cols=feature_cols,
                    split_strategy=split_strategy,
                    feature_set=feature_set,
                    run_label=run_label,
                )

            except Exception as exc:
                logger.error("[%s] Falló: %s", model_name, exc, exc_info=True)
                with mlflow.start_run(run_name=model_name, nested=True):
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", str(exc)[:250])

        if not results:
            logger.error("Ningún modelo completó el entrenamiento.")
            mlflow.set_tag("status", "ALL_FAILED")
            return parent_run.info.run_id, {}, {}

        # ── Métricas comparativas en el run padre ─────────────────────────────
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

    return parent_run.info.run_id, results, cv_aucs


def main() -> None:
    """Entrypoint: carga splits → construye conjuntos de features → entrena → MLflow."""
    args        = parse_args()
    model_names = args.models or DEFAULT_MODELS
    data_path   = Path(args.data_dir)

    df_train = pd.read_parquet(data_path / "train.parquet")
    df_test  = pd.read_parquet(data_path / "test.parquet")
    df_oot   = pd.read_parquet(data_path / "oot.parquet")

    with open(data_path / "feature_cols.json") as fh:
        feature_cols_all: list[str] = json.load(fh)

    # ── Determinar los conjuntos de features a entrenar ───────────────────────
    sets_to_run: list[tuple[str, list[str]]] = []
    if args.feature_set in ("A", "both"):
        sets_to_run.append(("A", feature_cols_all))
    if args.feature_set in ("B", "both"):
        sets_to_run.append(("B", _filter_option_b(feature_cols_all)))

    is_both = len(sets_to_run) > 1

    logger.info("=" * 70)
    logger.info("DEBIT RECURRENCE CLASSIFIER — ENTRENAMIENTO MULTI-MODELO")
    logger.info("=" * 70)
    logger.info("Modelos:        %s", model_names)
    logger.info("Split strategy: %s", args.split_strategy)
    logger.info("Feature set:    %s", args.feature_set)
    logger.info("Features (A):   %d | train=%d | test=%d | oot=%d",
                len(feature_cols_all), len(df_train), len(df_test), len(df_oot))
    if is_both:
        logger.info("Features (B):   %d", len(sets_to_run[1][1]))
    logger.info("n_trials:       %d", args.n_trials)

    scale_pos_weight = _compute_scale_pos_weight(df_train[TARGET_COL].astype(int))
    logger.info("scale_pos_weight=%.3f  (desbalance S8)", scale_pos_weight)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Un parent run por feature_set ─────────────────────────────────────────
    for fs_label, fs_cols in sets_to_run:
        pkl_suffix = f"_{fs_label}" if is_both else ""
        parent_run_id, results, cv_aucs = _run_feature_set(
            model_names=model_names,
            df_train=df_train,
            df_test=df_test,
            df_oot=df_oot,
            feature_cols=fs_cols,
            feature_set=fs_label,
            split_strategy=args.split_strategy,
            n_trials=args.n_trials,
            data_path=data_path,
            pkl_suffix=pkl_suffix,
            scale_pos_weight=scale_pos_weight,
        )

        if not results:
            continue

        # ── Resumen en log por feature_set ────────────────────────────────────
        best_name = max(results, key=lambda m: results[m]["test_auc"])
        run_label = _build_run_label(args.split_strategy, fs_label)
        logger.info("=" * 70)
        logger.info("RESUMEN — %s", run_label.upper())
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
            args.mlflow_uri, EXPERIMENT_NAME, parent_run_id,
        )


if __name__ == "__main__":
    main()
