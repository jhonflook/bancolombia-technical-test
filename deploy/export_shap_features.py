#!/usr/bin/env python3
"""
Calcula SHAP values para los modelos entrenados y guarda el ranking de importancia
media absoluta en debit_model_features para consumo en Metabase.

Soporta todos los modelos del MODEL_REGISTRY:
  - xgboost, gradient_boosting, random_forest → shap.TreeExplainer
  - logistic_regression                        → shap.LinearExplainer (con escalado)

Por defecto procesa los tres modelos de producción (xgboost, gradient_boosting,
logistic_regression). Los resultados reemplazan las feature importances nativas
en debit_model_features, dando una vista unificada SHAP en Metabase.

Uso:
    uv run python deploy/export_shap_features.py                          # todos
    uv run python deploy/export_shap_features.py --model-name xgboost
    uv run python deploy/export_shap_features.py --top-features 30 --sample 5000
    uv run python deploy/export_shap_features.py --overwrite
"""

import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import shap
from sqlalchemy import text

from src.database.connections import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR  = "data/artifacts"
TRAIN_PATH     = f"{ARTIFACTS_DIR}/train.parquet"

_TREE_MODELS   = ("xgboost", "gradient_boosting", "random_forest")
_LINEAR_MODELS = ("logistic_regression",)
_DEFAULT_MODELS = ["xgboost", "gradient_boosting", "logistic_regression"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta SHAP features de modelos a debitdb")
    p.add_argument(
        "--model-name",
        default="all",
        choices=["all", "xgboost", "gradient_boosting", "logistic_regression", "random_forest"],
        help="Modelo a procesar. 'all' itera sobre los tres modelos de producción (default)",
    )
    p.add_argument("--train-path",   default=TRAIN_PATH)
    p.add_argument("--top-features", type=int, default=50)
    p.add_argument(
        "--sample", type=int, default=3000,
        help="Filas de train usadas para el cálculo SHAP (0 = todas)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Reemplaza registros existentes del run en debit_model_features",
    )
    p.add_argument(
        "--feature-set",
        default="A",
        choices=["A", "B"],
        help=(
            "Conjunto de features del run a procesar (A = todas, B = sin pago/excedentes/canales). "
            "Selecciona el PKL y el run_id correspondientes. Default: 'A'"
        ),
    )
    return p.parse_args()


def _get_run_info(model_name: str, feature_set: str = "A") -> tuple[str, str] | None:
    """Devuelve (run_id, split_strategy) más reciente para el modelo y feature_set dados."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT run_id, split_strategy FROM debit_model_metrics "
                "WHERE model_name = :mn AND feature_set = :fs "
                "ORDER BY run_date DESC LIMIT 1"
            ),
            {"mn": model_name, "fs": feature_set},
        ).fetchone()
    return (row[0], row[1]) if row else None


def _compute_shap(
    clf,
    model_name: str,
    X_raw: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int,
) -> np.ndarray:
    """Devuelve mean-abs SHAP values de shape (n_features,) sobre una muestra de X."""
    if sample_n and sample_n < len(X_raw):
        X_bg = X_raw.sample(n=sample_n, random_state=42)
    else:
        X_bg = X_raw.copy()

    inner_model = clf.model

    if model_name in _TREE_MODELS:
        logger.info("[%s] TreeExplainer sobre %d muestras…", model_name, len(X_bg))
        try:
            explainer   = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_bg, check_additivity=False)
        except Exception as exc:
            logger.warning("[%s] TreeExplainer falló (%s). Usando PermutationExplainer…", model_name, exc)
            background  = shap.sample(X_bg, min(200, len(X_bg)))
            explainer   = shap.Explainer(inner_model.predict_proba, background)
            shap_out    = explainer(X_bg)
            shap_values = shap_out.values[:, :, 1]

    elif model_name in _LINEAR_MODELS:
        logger.info("[%s] LinearExplainer sobre %d muestras…", model_name, len(X_bg))
        scaler = getattr(clf, "scaler", None)
        if scaler is not None:
            X_scaled = pd.DataFrame(
                scaler.transform(X_bg),
                columns=feature_cols,
                index=X_bg.index,
            )
        else:
            X_scaled = X_bg
        explainer   = shap.LinearExplainer(inner_model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

    else:
        raise ValueError(f"Modelo no soportado para SHAP: {model_name}")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.abs(shap_values).mean(axis=0)


def _process_model(
    model_name: str,
    train_path: str,
    top_features: int,
    sample_n: int,
    overwrite: bool,
    feature_set: str = "A",
) -> bool:
    """Calcula SHAP para un modelo y escribe en debit_model_features. Devuelve True si OK."""
    # PKL con sufijo de feature_set (ej. model_xgboost_A.pkl).
    # Retroceso a model_{name}.pkl para backward compat con runs anteriores a la parametrización.
    import os
    pkl_path = f"{ARTIFACTS_DIR}/model_{model_name}_{feature_set}.pkl"
    if not os.path.exists(pkl_path) and feature_set == "A":
        pkl_path = f"{ARTIFACTS_DIR}/model_{model_name}.pkl"

    run_info = _get_run_info(model_name, feature_set)
    if run_info is None:
        logger.warning(
            "[%s] No se encontró run en debit_model_metrics para feature_set=%s. "
            "Ejecuta primero: make export-model-metrics",
            model_name, feature_set,
        )
        return False

    run_id, split_strategy = run_info
    logger.info("[%s] run_id=%s | split_strategy=%s", model_name, run_id[:8], split_strategy)

    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM debit_model_features WHERE run_id = :rid LIMIT 1"),
            {"rid": run_id},
        ).fetchone()

    if exists and not overwrite:
        logger.info(
            "[%s] Ya existen features SHAP para run_id=%s. Usa --overwrite para reemplazar.",
            model_name, run_id[:8],
        )
        return True

    logger.info("[%s] Cargando modelo desde %s…", model_name, pkl_path)
    try:
        with open(pkl_path, "rb") as f:
            artifact = pickle.load(f)
    except FileNotFoundError:
        logger.error("[%s] PKL no encontrado: %s", model_name, pkl_path)
        return False

    clf          = artifact["model"]
    feature_cols = artifact["feature_cols"]

    logger.info("[%s] Cargando train desde %s…", model_name, train_path)
    train = pd.read_parquet(train_path, columns=["var_rta"] + feature_cols)
    X     = train[feature_cols].fillna(0.0)

    mean_abs_shap = _compute_shap(clf, model_name, X, feature_cols, sample_n)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": mean_abs_shap})
        .sort_values("importance", ascending=False)
        .head(top_features)
        .reset_index(drop=True)
    )
    importance_df["rank"] = importance_df.index + 1
    logger.info(
        "[%s] Top 5 features por SHAP:\n%s",
        model_name, importance_df.head(5).to_string(index=False),
    )

    rows = [
        {
            "run_id":         run_id,
            "model_name":     model_name,
            "feature_name":   row["feature"],
            "importance":     float(row["importance"]),
            "rank":           int(row["rank"]),
            "feature_set":    feature_set,
            "split_strategy": split_strategy,
        }
        for _, row in importance_df.iterrows()
    ]

    with engine.begin() as conn:
        if exists:
            conn.execute(
                text("DELETE FROM debit_model_features WHERE run_id = :rid"),
                {"rid": run_id},
            )
        pd.DataFrame(rows).to_sql(
            "debit_model_features", conn, if_exists="append", index=False
        )

    logger.info(
        "[%s] Escritas %d filas SHAP en debit_model_features (run=%s).",
        model_name, len(rows), run_id[:8],
    )
    return True


def main() -> None:
    args = _parse_args()

    model_names = _DEFAULT_MODELS if args.model_name == "all" else [args.model_name]

    logger.info("Feature set: %s | Modelos: %s", args.feature_set, model_names)
    success, failed = [], []
    for name in model_names:
        ok = _process_model(
            model_name=name,
            train_path=args.train_path,
            top_features=args.top_features,
            sample_n=args.sample,
            overwrite=args.overwrite,
            feature_set=args.feature_set,
        )
        (success if ok else failed).append(name)

    logger.info(
        "Completado [feature_set=%s]: %d OK %s | %d fallidos %s",
        args.feature_set, len(success), success, len(failed), failed,
    )


if __name__ == "__main__":
    main()
