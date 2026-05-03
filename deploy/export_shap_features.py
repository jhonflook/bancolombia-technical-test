#!/usr/bin/env python3
"""
Calcula SHAP values para el modelo gradient_boosting (HistGBM) y guarda el ranking
de importancia media absoluta en debit_model_features para consumo en Metabase.

Uso:
    uv run python deploy/export_shap_features.py
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

MODEL_NAME = "gradient_boosting"
PKL_PATH   = "data/artifacts/model_gradient_boosting.pkl"
TRAIN_PATH = "data/artifacts/train.parquet"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta SHAP features de gradient_boosting a debitdb")
    p.add_argument("--pkl-path",     default=PKL_PATH)
    p.add_argument("--train-path",   default=TRAIN_PATH)
    p.add_argument("--top-features", type=int, default=50)
    p.add_argument("--sample",       type=int, default=3000,
                   help="Filas de train usadas para el cálculo SHAP (0 = todas)")
    p.add_argument("--overwrite",    action="store_true",
                   help="Reemplaza registros existentes del run gradient_boosting")
    return p.parse_args()


def _get_gb_run_id() -> str | None:
    """Devuelve el run_id más reciente de gradient_boosting en debit_model_metrics."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT run_id FROM debit_model_metrics "
                "WHERE model_name = :mn "
                "ORDER BY run_date DESC LIMIT 1"
            ),
            {"mn": MODEL_NAME},
        ).fetchone()
    return row[0] if row else None


def _compute_shap(model, X: pd.DataFrame, sample_n: int) -> np.ndarray:
    """Devuelve mean-abs SHAP values shape (n_features,) sobre una muestra de X."""
    if sample_n and sample_n < len(X):
        X_bg = X.sample(n=sample_n, random_state=42)
    else:
        X_bg = X

    logger.info("Calculando SHAP con TreeExplainer sobre %d muestras…", len(X_bg))
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_bg, check_additivity=False)
        # HistGBM devuelve (n_samples, n_features) para clase-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as exc:
        logger.warning("TreeExplainer falló (%s). Usando PermutationExplainer…", exc)
        background  = shap.sample(X_bg, min(200, len(X_bg)))
        explainer   = shap.Explainer(model.predict_proba, background)
        shap_out    = explainer(X_bg)
        # predict_proba → shape (n_samples, n_features, 2); tomar clase-1
        shap_values = shap_out.values[:, :, 1]

    return np.abs(shap_values).mean(axis=0)


def main() -> None:
    args = _parse_args()

    run_id = _get_gb_run_id()
    if run_id is None:
        logger.error(
            "No se encontró run de '%s' en debit_model_metrics. "
            "Ejecuta primero: make export-model-metrics",
            MODEL_NAME,
        )
        return

    logger.info("run_id gradient_boosting: %s", run_id[:8])

    # ── Verificar si ya existen y si se debe sobrescribir ──────────────────────
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM debit_model_features WHERE run_id = :rid LIMIT 1"),
            {"rid": run_id},
        ).fetchone()

    if exists and not args.overwrite:
        logger.info(
            "Ya existen features SHAP para run_id=%s. Usa --overwrite para reemplazar.",
            run_id[:8],
        )
        return

    # ── Cargar modelo y datos ──────────────────────────────────────────────────
    logger.info("Cargando modelo desde %s…", args.pkl_path)
    with open(args.pkl_path, "rb") as f:
        artifact = pickle.load(f)

    clf          = artifact["model"]
    feature_cols = artifact["feature_cols"]
    hist_gbm     = clf.model  # HistGradientBoostingClassifier

    logger.info("Cargando train desde %s…", args.train_path)
    train = pd.read_parquet(args.train_path, columns=["var_rta"] + feature_cols)
    X     = train[feature_cols]

    # ── Calcular SHAP ──────────────────────────────────────────────────────────
    mean_abs_shap = _compute_shap(hist_gbm, X, args.sample)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": mean_abs_shap})
        .sort_values("importance", ascending=False)
        .head(args.top_features)
        .reset_index(drop=True)
    )
    importance_df["rank"] = importance_df.index + 1
    logger.info("Top 5 features por SHAP:\n%s", importance_df.head(5).to_string(index=False))

    # ── Upsert en debit_model_features ────────────────────────────────────────
    rows = [
        {
            "run_id":       run_id,
            "model_name":   MODEL_NAME,
            "feature_name": row["feature"],
            "importance":   float(row["importance"]),
            "rank":         int(row["rank"]),
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
        "Escritas %d filas SHAP en debit_model_features para gradient_boosting (run=%s).",
        len(rows), run_id[:8],
    )


if __name__ == "__main__":
    main()
