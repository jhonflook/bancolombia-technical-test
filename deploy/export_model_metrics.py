#!/usr/bin/env python3
"""
Lee los runs hijos de MLflow del experimento 'debit-models' y escribe
las métricas y feature importances en debitdb para consumo en Metabase.

Tablas destino:
  debit_model_metrics  — una fila por run_id (modelo × ejecución)
  debit_model_features — una fila por (run_id, feature), top-50 por importancia

Uso:
    uv run python deploy/export_model_metrics.py
    uv run python deploy/export_model_metrics.py --mlflow-uri http://localhost:5000
    uv run python deploy/export_model_metrics.py --experiment debit-models --top-features 30
"""

import argparse
import json
import logging
from datetime import timezone

import mlflow
import pandas as pd
from sqlalchemy import text

from src.database.connections import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "debit-models"

# Mapeo métrica MLflow → columna en debit_model_metrics
_METRIC_MAP = {
    "cv_auc":              "cv_auc",
    "train_auc_roc":       "train_auc",
    "train_ks":            "train_ks",
    "train_f1_05":         "train_f1",
    "train_auc_pr":        "train_auc_pr",
    "train_precision_05":  "train_precision",
    "train_recall_05":     "train_recall",
    "test_auc_roc":        "test_auc",
    "test_ks":             "test_ks",
    "test_f1_05":          "test_f1",
    "test_auc_pr":         "test_auc_pr",
    "test_precision_05":   "test_precision",
    "test_recall_05":      "test_recall",
    "oot_auc_roc":         "oot_auc",
    "oot_ks":              "oot_ks",
    "oot_f1_05":           "oot_f1",
    "oot_auc_pr":          "oot_auc_pr",
    "oot_precision_05":    "oot_precision",
    "oot_recall_05":       "oot_recall",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta métricas de MLflow a debitdb")
    p.add_argument("--mlflow-uri",   default="http://localhost:5000")
    p.add_argument("--experiment",   default=EXPERIMENT_NAME)
    p.add_argument("--top-features", type=int, default=50,
                   help="Máximo de features a guardar por run (por importancia)")
    p.add_argument("--overwrite",    action="store_true",
                   help="Reemplaza registros existentes con el mismo run_id")
    return p.parse_args()


def _fetch_child_runs(client: mlflow.MlflowClient, experiment_id: str) -> list:
    """Devuelve todos los runs hijos (modelos individuales) del experimento."""
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId != ''",
        max_results=500,
    )
    # Excluir runs con status FAILED
    return [r for r in all_runs if r.info.status == "FINISHED"]


def _extract_metrics(run) -> dict:
    metrics = run.data.metrics
    row: dict = {}
    for mlflow_key, col in _METRIC_MAP.items():
        row[col] = metrics.get(mlflow_key)
    return row


def _extract_params(run) -> dict:
    params = run.data.params
    tags   = run.data.tags
    best_params: dict = {}
    skip = {"model_type", "n_features", "n_trials", "scale_pos_weight", "split_strategy", "feature_set"}
    for k, v in params.items():
        if k not in skip:
            try:
                best_params[k] = float(v)
            except (ValueError, TypeError):
                best_params[k] = v
    return {
        "model_name":       params.get("model_type", tags.get("model_type", "")),
        "n_features":       int(params["n_features"]) if "n_features" in params else None,
        "n_trials":         int(params["n_trials"])   if "n_trials"   in params else None,
        "scale_pos_weight": float(params["scale_pos_weight"]) if "scale_pos_weight" in params else None,
        "best_params":      json.dumps(best_params) if best_params else None,
        # n_train/n_test/n_oot se loguean como tags, no como params
        "train_size":       int(tags["n_train"]) if "n_train" in tags else None,
        "test_size":        int(tags["n_test"])  if "n_test"  in tags else None,
        "oot_size":         int(tags["n_oot"])   if "n_oot"   in tags else None,
    }


def _extract_split_strategy(run, client: mlflow.MlflowClient) -> str:
    """Busca split_strategy en tags del run hijo o, en su defecto, en el run padre."""
    # El run hijo lo tiene en tags desde la migración de parametrización
    tag_val = run.data.tags.get("split_strategy")
    if tag_val:
        return tag_val
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if parent_id:
        try:
            parent = client.get_run(parent_id)
            return parent.data.params.get("split_strategy", "unknown")
        except Exception:
            pass
    return "unknown"


def _extract_feature_set(run) -> str:
    """Extrae feature_set del tag del run hijo (A | B)."""
    return run.data.tags.get("feature_set", "A")


def _fetch_feature_importance(
    run, client: mlflow.MlflowClient, top_n: int
) -> list[dict]:
    """Descarga feature_importance CSV desde los artefactos del run."""
    try:
        artifacts = client.list_artifacts(run.info.run_id, path="feature_importance")
        if not artifacts:
            return []
        artifact_uri = f"{run.info.artifact_uri}/{artifacts[0].path}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        df = pd.read_csv(local_path)
        df = df.nlargest(top_n, "importance").reset_index(drop=True)
        return [
            {
                "run_id":       run.info.run_id,
                "model_name":   run.data.params.get("model_type", ""),
                "feature_name": row["feature"],
                "importance":   float(row["importance"]),
                "rank":         idx + 1,
            }
            for idx, row in df.iterrows()
        ]
    except Exception as exc:
        logger.warning("[%s] No se pudo leer feature importance: %s",
                       run.info.run_id[:8], exc)
        return []


def _upsert_metrics(rows: list[dict], overwrite: bool) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    written = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            run_id = row["run_id"]
            exists = conn.execute(
                text("SELECT 1 FROM debit_model_metrics WHERE run_id = :rid"),
                {"rid": run_id},
            ).fetchone()
            if exists and not overwrite:
                logger.info("run_id=%s ya existe, omitiendo (usa --overwrite para reemplazar).", run_id[:8])
                continue
            if exists:
                conn.execute(
                    text("DELETE FROM debit_model_metrics WHERE run_id = :rid"),
                    {"rid": run_id},
                )
            cols   = ", ".join(row.index)
            params = ", ".join(f":{c}" for c in row.index)
            conn.execute(
                text(f"INSERT INTO debit_model_metrics ({cols}) VALUES ({params})"),
                row.to_dict(),
            )
            written += 1
    return written


def _upsert_features(rows: list[dict], overwrite: bool) -> int:
    if not rows:
        return 0
    run_ids = list({r["run_id"] for r in rows})
    with engine.begin() as conn:
        for rid in run_ids:
            exists = conn.execute(
                text("SELECT 1 FROM debit_model_features WHERE run_id = :rid LIMIT 1"),
                {"rid": rid},
            ).fetchone()
            if exists and not overwrite:
                continue
            if exists:
                conn.execute(
                    text("DELETE FROM debit_model_features WHERE run_id = :rid"),
                    {"rid": rid},
                )
        df = pd.DataFrame(rows)
        df.to_sql("debit_model_features", conn, if_exists="append", index=False)
    return len(rows)


def main() -> None:
    args = _parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.MlflowClient(args.mlflow_uri)

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        logger.error("Experimento '%s' no encontrado en MLflow.", args.experiment)
        return

    logger.info("Experimento: %s (id=%s)", args.experiment, experiment.experiment_id)
    child_runs = _fetch_child_runs(client, experiment.experiment_id)
    logger.info("%d runs hijos encontrados.", len(child_runs))

    metric_rows:  list[dict] = []
    feature_rows: list[dict] = []

    for run in child_runs:
        rid         = run.info.run_id
        params      = _extract_params(run)
        metrics     = _extract_metrics(run)
        split       = _extract_split_strategy(run, client)
        feature_set = _extract_feature_set(run)
        parent      = run.data.tags.get("mlflow.parentRunId")

        run_date = pd.Timestamp(run.info.start_time, unit="ms", tz=timezone.utc)

        row = {
            "run_id":          rid,
            "parent_run_id":   parent,
            "experiment_name": args.experiment,
            "split_strategy":  split,
            "feature_set":     feature_set,
            "run_date":        run_date,
            **params,
            **metrics,
        }
        metric_rows.append(row)

        feat_rows = _fetch_feature_importance(run, client, args.top_features)
        for fr in feat_rows:
            fr["split_strategy"] = split
            fr["feature_set"]    = feature_set
        feature_rows.extend(feat_rows)

        logger.info("Procesado run %s | %s | split=%s | feature_set=%s | test_auc=%.4f",
                    rid[:8], params["model_name"], split, feature_set,
                    metrics.get("test_auc") or 0)

    n_metrics  = _upsert_metrics(metric_rows, args.overwrite)
    n_features = _upsert_features(feature_rows, args.overwrite)
    logger.info("Escritos: %d filas en debit_model_metrics | %d filas en debit_model_features",
                n_metrics, n_features)


if __name__ == "__main__":
    main()
