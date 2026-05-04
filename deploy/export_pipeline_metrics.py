#!/usr/bin/env python3
"""
Recopila y persiste métricas de preprocesamiento del pipeline en debit_pipeline_metrics.

Cubre cuatro etapas:
  1. loader             — dimensiones y calidad de carga por fuente CSV
  2. build_model        — modelo analítico: filas, columnas, balance de clases
  3. feature_engineering — dimensiones y calidad de cada partición (train/test/oot)
  4. feature_selection  — funnel de reducción por paso y distribución por grupo temático

Fuentes de datos:
  - CSVs del datalake (solo KEY_COLS para eficiencia)   → métricas loader
  - Tablas postgres debit_*                             → filas cargadas por tabla
  - data/artifacts/analytical_model.parquet             → métricas build_model
  - data/artifacts/{train,test,oot}.parquet             → métricas feature_engineering
  - data/artifacts/feature_selection_metrics.json       → métricas feature_selection
  - data/artifacts/feature_cols.json                    → features finales seleccionadas

Uso:
    uv run python deploy/export_pipeline_metrics.py
    uv run python deploy/export_pipeline_metrics.py --split-strategy random --overwrite
    uv run python deploy/export_pipeline_metrics.py --datalake-path ./datalake
"""

import argparse
import datetime
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.database.connections import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("data/artifacts")
DATALAKE_DIR  = Path("./datalake")

KEY_COLS = ["num_doc", "obl17", "f_analisis"]
TARGET_COL = "var_rta"

_CSV_MAP = {
    "clientes":   "clientes_seleccionados_prueba.csv",
    "excedentes": "debitos_dev_cruce_excedente_poblacion_prueba_tecnica.csv",
    "gestiones":  "debitos_dev_cruce_gestiones_poblacion_prueba_tecnica.csv",
    "moras":      "debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv",
    "pagos":      "debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv",
    "canales":    "debitos_dev_cruce_canales_poblacion_sel_vars_prueba_tecnica_1.csv",
}

_TABLE_MAP = {
    "clientes":   "debit_clients",
    "excedentes": "debit_excedentes",
    "gestiones":  "debit_gestiones",
    "moras":      "debit_moras",
    "pagos":      "debit_pagos",
    "canales":    "debit_canales",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta métricas de preprocesamiento a debitdb")
    p.add_argument(
        "--datalake-path", default=str(DATALAKE_DIR),
        help="Ruta al directorio con los CSVs fuente (default: ./datalake)",
    )
    p.add_argument(
        "--artifacts-dir", default=str(ARTIFACTS_DIR),
        help="Ruta al directorio de artefactos (default: data/artifacts)",
    )
    p.add_argument(
        "--split-strategy", default="random", choices=["temporal", "random"],
        help="Estrategia de split usada para generar los parquets (default: random)",
    )
    p.add_argument(
        "--run-id", default=None,
        help="ID del run (default: auto-generado desde el mtime de train.parquet)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Eliminar métricas previas del mismo run_id antes de insertar",
    )
    return p.parse_args()


def _make_run_id(artifacts: Path, split_strategy: str) -> str:
    """Generar run_id estable basado en el mtime de train.parquet."""
    train = artifacts / "train.parquet"
    if train.exists():
        mtime = train.stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%dT%H%M%S")
        return f"pipeline_{split_strategy}_{ts}"
    return f"pipeline_{split_strategy}_{uuid.uuid4().hex[:8]}"


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _write_metrics(rows: list[dict], run_id: str, split_strategy: str, overwrite: bool) -> None:
    if not rows:
        return
    df = _rows_to_df(rows)
    with engine.begin() as conn:
        if overwrite:
            # Borrar TODAS las filas de esta estrategia (no solo el run_id)
            # para que re-ejecutar el mismo split_strategy reemplace la observación anterior.
            conn.execute(
                text("DELETE FROM debit_pipeline_metrics WHERE split_strategy = :ss"),
                {"ss": split_strategy},
            )
        df.to_sql("debit_pipeline_metrics", conn, if_exists="append", index=False)
    logger.info(
        "Escritas %d métricas | run_id=%s | split_strategy=%s",
        len(rows), run_id[:20], split_strategy,
    )


def _make_row(run_id, run_date, split_strategy, stage, source_name, metric_name, metric_value):
    return {
        "run_id":         run_id,
        "run_date":       run_date,
        "split_strategy": split_strategy,
        "stage":          stage,
        "source_name":    source_name,
        "metric_name":    metric_name,
        "metric_value":   float(metric_value),
    }


# ─── Etapa 1: loader ──────────────────────────────────────────────────────────

def _csv_loader_metrics(
    datalake: Path, split_strategy: str, run_id: str, run_date: datetime.datetime,
) -> list[dict]:
    """Leer solo KEY_COLS de cada CSV para obtener conteos y estadísticas de duplicados."""
    rows = []
    for alias, fname in _CSV_MAP.items():
        csv_path = datalake / fname
        if not csv_path.exists():
            logger.warning("[loader] CSV no encontrado: %s — omitiendo", csv_path)
            continue

        # Leer KEY_COLS + todas las columnas solo para conteo de cols
        logger.info("[loader] Escaneando %s…", alias)
        try:
            # Conteo total de columnas sin cargar todas las filas (para CSVs grandes)
            header = pd.read_csv(csv_path, nrows=0)
            cols_count = len(header.columns)

            # Leer solo KEY_COLS para estadísticas de dedup (muy ligero)
            df_keys = pd.read_csv(csv_path, usecols=KEY_COLS, low_memory=False)
            df_keys["f_analisis"] = pd.to_datetime(df_keys["f_analisis"]).dt.date

            rows_raw = len(df_keys)

            # Duplicados exactos sobre KEY_COLS (proxy de filas exactamente iguales)
            dup_exact = int(df_keys.duplicated(subset=KEY_COLS, keep=False).sum() // 2)

            # Tras dedup: claves que siguen duplicadas = conflictos de valores
            df_dedup = df_keys.drop_duplicates(subset=KEY_COLS)
            rows_after_dedup = len(df_dedup)
            conflict_mask = df_dedup.duplicated(subset=KEY_COLS, keep=False)
            dup_conflict = int(conflict_mask.sum())
            rows_excluded = dup_conflict  # todas las filas con clave conflictiva se excluyen
            rows_loaded_est = rows_after_dedup - rows_excluded

        except Exception as exc:
            logger.warning("[loader] Error escaneando %s: %s", alias, exc)
            continue

        # Filas reales cargadas desde postgres (fuente de verdad)
        try:
            table = _TABLE_MAP[alias]
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                rows_loaded_db = int(result.scalar())
        except Exception:
            rows_loaded_db = rows_loaded_est

        base = {"run_id": run_id, "run_date": run_date,
                "split_strategy": split_strategy, "stage": "loader", "source_name": alias}

        for metric, val in [
            ("rows_raw",           rows_raw),
            ("rows_after_dedup",   rows_after_dedup),
            ("rows_excluded",      rows_excluded),
            ("dup_exact_count",    dup_exact),
            ("dup_conflict_count", dup_conflict),
            ("rows_loaded",        rows_loaded_db),
            ("cols_count",         cols_count),
        ]:
            rows.append({**base, "metric_name": metric, "metric_value": float(val)})

        logger.info(
            "[loader] %s: raw=%d → dedup=%d → excluidas=%d → cargadas=%d | cols=%d",
            alias, rows_raw, rows_after_dedup, rows_excluded, rows_loaded_db, cols_count,
        )

    return rows


# ─── Etapa 2: build_model ─────────────────────────────────────────────────────

def _build_model_metrics(
    artifacts: Path, split_strategy: str, run_id: str, run_date: datetime.datetime,
) -> list[dict]:
    """Métricas del modelo analítico (JOIN de todas las fuentes)."""
    model_path = artifacts / "analytical_model.parquet"
    if not model_path.exists():
        logger.warning("[build_model] analytical_model.parquet no encontrado — omitiendo")
        return []

    logger.info("[build_model] Leyendo metadata de analytical_model.parquet…")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(model_path)
    meta = pf.metadata
    rows_total = meta.num_rows
    cols_total = meta.num_columns
    file_mb    = model_path.stat().st_size / 1024**2

    # Solo leer TARGET_COL para balance de clases (evita OOM con ~300 MB en pandas)
    df_target = pf.read(columns=[TARGET_COL]).to_pandas()
    class1  = int(df_target[TARGET_COL].sum())
    class0  = rows_total - class1
    balance = round(class0 / class1, 4) if class1 else 0.0
    del df_target

    # Cobertura de JOINs: inferida desde los nombres de columnas (sin leer datos)
    schema_names = set(meta.schema.names)
    join_coverage = {
        "excedentes": 100.0 if any("excedente" in c for c in schema_names) else 0.0,
        "gestiones":  100.0 if any("gestiones" in c for c in schema_names) else 0.0,
        "moras":      100.0 if any("mora" in c for c in schema_names) else 0.0,
        "pagos":      100.0 if any("pago_debito" in c for c in schema_names) else 0.0,
        "canales":    100.0 if any("trx_" in c for c in schema_names) else 0.0,
    }

    base = {"run_id": run_id, "run_date": run_date,
            "split_strategy": split_strategy, "stage": "build_model", "source_name": "analytical"}

    result = []
    for metric, val in [
        ("rows_total",     rows_total),
        ("cols_total",     cols_total),
        ("file_size_mb",   round(file_mb, 2)),
        ("class1_count",   class1),
        ("class0_count",   class0),
        ("balance_ratio",  balance),
        ("target_rate",    round(class1 / rows_total, 4)),
    ]:
        result.append({**base, "metric_name": metric, "metric_value": float(val)})

    for src, pct in join_coverage.items():
        result.append({**base, "metric_name": f"join_coverage_{src}_pct", "metric_value": round(pct, 2)})

    logger.info(
        "[build_model] %d filas × %d cols | clase0=%d, clase1=%d | %.2f MB (archivo)",
        rows_total, cols_total, class0, class1, file_mb,
    )
    return result


# ─── Etapa 3: feature_engineering ────────────────────────────────────────────

def _feature_engineering_metrics(
    artifacts: Path, split_strategy: str, run_id: str, run_date: datetime.datetime,
) -> list[dict]:
    """Métricas de calidad y dimensiones para cada partición (train/test/oot)."""
    feat_cols_path = artifacts / "feature_cols.json"
    n_selected = 0
    if feat_cols_path.exists():
        with open(feat_cols_path) as f:
            n_selected = len(json.load(f))

    result = []
    for partition in ("train", "test", "oot"):
        path = artifacts / f"{partition}.parquet"
        if not path.exists():
            continue

        logger.info("[feature_engineering] Leyendo metadata de %s.parquet…", partition)
        import pyarrow.parquet as pq
        pf        = pq.ParquetFile(path)
        meta      = pf.metadata
        rows      = meta.num_rows
        all_cols  = meta.schema.names
        feature_candidates = len([c for c in all_cols if c not in (KEY_COLS + [TARGET_COL])])
        file_mb   = path.stat().st_size / 1024**2

        # Clase solo requiere leer TARGET_COL
        df_tgt    = pf.read(columns=[TARGET_COL]).to_pandas()
        class1    = int(df_tgt[TARGET_COL].sum())
        class0    = rows - class1
        target_rate = class1 / rows if rows else 0.0
        del df_tgt

        # Nulos e infinitos: leer solo columnas de features (no key cols ni target)
        feat_cols_read = [c for c in all_cols if c not in (KEY_COLS + [TARGET_COL])]
        df_feat   = pf.read(columns=feat_cols_read).to_pandas()
        null_count = int(df_feat.isnull().sum().sum())
        inf_count  = int(np.isinf(df_feat.select_dtypes(include="number").values).sum())
        memory_mb  = df_feat.memory_usage(deep=False).sum() / 1024**2
        del df_feat

        base = {"run_id": run_id, "run_date": run_date,
                "split_strategy": split_strategy,
                "stage": "feature_engineering", "source_name": partition}

        for metric, val in [
            ("rows",             rows),
            ("cols_candidate",   feature_candidates),
            ("cols_selected",    n_selected),
            ("memory_mb",        round(memory_mb, 2)),
            ("file_size_mb",     round(file_mb, 2)),
            ("class1_count",     class1),
            ("class0_count",     class0),
            ("target_rate",      round(target_rate, 4)),
            ("null_count",       null_count),
            ("inf_count",        inf_count),
        ]:
            result.append({**base, "metric_name": metric, "metric_value": float(val)})

        logger.info(
            "[feature_engineering] %s: %d × %d | clase0=%d, clase1=%d | nulos=%d | inf=%d | %.2f MB",
            partition, rows, feature_candidates, class0, class1, null_count, inf_count, memory_mb,
        )

    return result


# ─── Etapa 4: feature_selection ───────────────────────────────────────────────

def _feature_selection_metrics(
    artifacts: Path, split_strategy: str, run_id: str, run_date: datetime.datetime,
) -> list[dict]:
    """Métricas del funnel de selección supervisada de features."""
    metrics_path = artifacts / "feature_selection_metrics.json"
    feat_cols_path = artifacts / "feature_cols.json"

    if not feat_cols_path.exists():
        logger.warning("[feature_selection] feature_cols.json no encontrado — omitiendo")
        return []

    with open(feat_cols_path) as f:
        final_features = json.load(f)

    base_overall = {
        "run_id": run_id, "run_date": run_date,
        "split_strategy": split_strategy,
        "stage": "feature_selection", "source_name": "overall",
    }

    result = []

    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)

        for metric in [
            "cols_initial", "cols_after_variance", "removed_variance_var",
            "removed_variance_sparse", "cols_after_anova", "cols_removed_anova",
            "cols_final", "cols_removed_elasticnet", "var_threshold",
            "anova_percentile", "elasticnet_l1", "elasticnet_c", "time_sec",
        ]:
            if metric in m:
                result.append({**base_overall, "metric_name": metric,
                                "metric_value": float(m[metric])})

        # Grupos temáticos de features finales
        if "groups" in m:
            for group_name, count in m["groups"].items():
                result.append({
                    "run_id": run_id, "run_date": run_date,
                    "split_strategy": split_strategy,
                    "stage": "feature_selection",
                    "source_name": group_name,
                    "metric_name": "cols_group",
                    "metric_value": float(count),
                })

        logger.info(
            "[feature_selection] Funnel: %d → %d (var) → %d (anova) → %d (elasticnet) | grupos=%s",
            m.get("cols_initial", "?"), m.get("cols_after_variance", "?"),
            m.get("cols_after_anova", "?"), m.get("cols_final", "?"),
            m.get("groups", {}),
        )
    else:
        # Fallback: solo conteo final desde feature_cols.json
        logger.info(
            "[feature_selection] feature_selection_metrics.json no encontrado — "
            "solo conteo final desde feature_cols.json (ejecuta select-features para métricas completas)"
        )
        result.append({**base_overall, "metric_name": "cols_final",
                        "metric_value": float(len(final_features))})

        # Clasificar grupos desde los nombres de features
        from src.dataset.feature_selection import _classify_feature_group
        groups: dict[str, int] = {}
        for feat in final_features:
            g = _classify_feature_group(feat)
            groups[g] = groups.get(g, 0) + 1

        for group_name, count in groups.items():
            result.append({
                "run_id": run_id, "run_date": run_date,
                "split_strategy": split_strategy,
                "stage": "feature_selection",
                "source_name": group_name,
                "metric_name": "cols_group",
                "metric_value": float(count),
            })

    return result


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    args     = _parse_args()
    artifacts = Path(args.artifacts_dir)
    datalake  = Path(args.datalake_path)
    run_date  = datetime.datetime.now(tz=datetime.timezone.utc)
    run_id    = args.run_id or _make_run_id(artifacts, args.split_strategy)

    logger.info("=" * 60)
    logger.info("EXPORT PIPELINE METRICS | run_id=%s", run_id)
    logger.info("split_strategy=%s | overwrite=%s", args.split_strategy, args.overwrite)
    logger.info("=" * 60)

    all_rows: list[dict] = []

    # Etapa 1 — loader (requiere CSVs)
    if datalake.exists():
        all_rows += _csv_loader_metrics(datalake, args.split_strategy, run_id, run_date)
    else:
        logger.warning("Datalake no encontrado (%s) — métricas de carga omitidas", datalake)

    # Etapa 2 — build_model
    all_rows += _build_model_metrics(artifacts, args.split_strategy, run_id, run_date)

    # Etapa 3 — feature_engineering
    all_rows += _feature_engineering_metrics(artifacts, args.split_strategy, run_id, run_date)

    # Etapa 4 — feature_selection
    all_rows += _feature_selection_metrics(artifacts, args.split_strategy, run_id, run_date)

    if not all_rows:
        logger.error("No se generaron métricas. Verifica las rutas de artefactos.")
        return

    _write_metrics(all_rows, run_id, args.split_strategy, args.overwrite)
    logger.info("Total métricas escritas: %d | run_id=%s", len(all_rows), run_id)


if __name__ == "__main__":
    main()
