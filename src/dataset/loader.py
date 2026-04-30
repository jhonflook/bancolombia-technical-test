"""CSV → PostgreSQL loader for the debit recurrent analysis project.

Reads the 6 source CSVs from datalake/, applies null imputation per
active assumptions (S4, S5), and bulk-inserts into the debit_* tables
using pandas + SQLAlchemy for chunk-based loading.

Usage:
    python -m src.dataset.loader [--chunk-size 5000] [--datalake-path ./datalake]

Assumptions applied:
  S4: canales absent rows (no transactional activity) → features JSONB = {}
  S5: excedentes nulls (young obligations) → kept as NULL (nullable columns)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.connections import engine

logger = logging.getLogger(__name__)

DATALAKE_FILES = {
    "clientes":   "clientes_seleccionados_prueba.csv",
    "excedentes": "debitos_dev_cruce_excedente_poblacion_prueba_tecnica.csv",
    "gestiones":  "debitos_dev_cruce_gestiones_poblacion_prueba_tecnica.csv",
    "moras":      "debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv",
    "pagos":      "debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv",
    "canales":    "debitos_dev_cruce_canales_poblacion_sel_vars_prueba_tecnica_1.csv",
}

KEY_COLS = ["num_doc", "obl17", "f_analisis"]
CANALES_SUMMARY_COLS = ["trx_mnt_total", "trx_mnt_total_smmlv", "trx_cnt_total"]


def _read_csv(path: Path, alias: str) -> pd.DataFrame:
    logger.info("Reading %s from %s", alias, path)
    df = pd.read_csv(path, low_memory=False)
    df["f_analisis"] = pd.to_datetime(df["f_analisis"]).dt.date
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.warning("  %s: dropped %d exact duplicate rows", alias, dropped)
    logger.info("  %s: %d rows × %d cols", alias, len(df), len(df.columns))
    return df


def _exclude_conflicting_keys(df: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Remove all rows whose composite key still appears more than once after exact dedup.

    After drop_duplicates(), remaining duplicates on KEY_COLS have the same key
    but different values — unresolvable without business context (strategy c).
    All rows in those groups are excluded to avoid introducing noise in features.
    """
    conflict_mask = df.duplicated(subset=KEY_COLS, keep=False)
    n_rows = conflict_mask.sum()
    if n_rows:
        n_groups = df[conflict_mask].groupby(KEY_COLS).ngroups
        logger.warning(
            "  %s: excluded %d rows across %d key groups with conflicting values (strategy c)",
            alias, n_rows, n_groups,
        )
        df = df[~conflict_mask].copy()
    return df


def _load_table(df: pd.DataFrame, table: str, chunk_size: int) -> None:
    total = len(df)
    loaded = 0
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        chunk.to_sql(table, engine, if_exists="append", index=False, method="multi")
        loaded += len(chunk)
        logger.info("  %s: %d / %d rows loaded", table, loaded, total)


def load_clientes(datalake: Path, chunk_size: int) -> None:
    df = _read_csv(datalake / DATALAKE_FILES["clientes"], "clientes")
    _load_table(df, "debit_clients", chunk_size)


def load_excedentes(datalake: Path, chunk_size: int) -> None:
    df = _read_csv(datalake / DATALAKE_FILES["excedentes"], "excedentes")
    # S5: nulls kept as NULL — nullable columns in the model
    _load_table(df, "debit_excedentes", chunk_size)


def load_gestiones(datalake: Path, chunk_size: int) -> None:
    df = _read_csv(datalake / DATALAKE_FILES["gestiones"], "gestiones")
    _load_table(df, "debit_gestiones", chunk_size)


def load_moras(datalake: Path, chunk_size: int) -> None:
    df = _read_csv(datalake / DATALAKE_FILES["moras"], "moras")
    _load_table(df, "debit_moras", chunk_size)


def load_pagos(datalake: Path, chunk_size: int) -> None:
    df = _read_csv(datalake / DATALAKE_FILES["pagos"], "pagos")
    _load_table(df, "debit_pagos", chunk_size)


def load_canales(datalake: Path, chunk_size: int) -> None:
    """Load canales: summary columns as typed, remaining as JSONB features dict.

    S4: rows absent from canales = no transactional activity.
    Those rows are not in the source file so no insert is needed;
    the LEFT JOIN in downstream queries will return NULL → impute 0.
    Dedup strategy c: 24 key groups with conflicting values are excluded entirely.
    """
    df = _read_csv(datalake / DATALAKE_FILES["canales"], "canales")
    df = _exclude_conflicting_keys(df, "canales")

    feature_cols = [c for c in df.columns if c not in KEY_COLS + CANALES_SUMMARY_COLS]

    # Build flat table: key + summary cols + features as JSON string.
    # psycopg2 cannot adapt raw Python dicts for JSONB; json.dumps() produces
    # a string that PostgreSQL parses and stores as JSONB automatically.
    rows = df[KEY_COLS + CANALES_SUMMARY_COLS].copy()
    feature_records = (
        df[feature_cols]
        .fillna(0)
        .infer_objects(copy=False)
        .to_dict(orient="records")
    )
    rows["features"] = [json.dumps(d) for d in feature_records]

    _load_table(rows, "debit_canales", chunk_size)


def _truncate_debit_tables() -> None:
    tables = [
        "debit_canales", "debit_pagos", "debit_moras",
        "debit_gestiones", "debit_excedentes", "debit_clients",
    ]
    with engine.begin() as conn:
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    logger.info("All debit tables truncated.")


def run(datalake_path: str = "./datalake", chunk_size: int = 5000, truncate: bool = False) -> None:
    """Execute full load pipeline for all 6 debit source files.

    Parameters
    ----------
    datalake_path:
        Path to the directory containing the 6 CSV files.
    chunk_size:
        Number of rows per insert batch.
    truncate:
        If True, truncate all debit tables before loading (idempotent re-run).
    """
    datalake = Path(datalake_path)

    if truncate:
        _truncate_debit_tables()

    load_clientes(datalake, chunk_size)
    load_excedentes(datalake, chunk_size)
    load_gestiones(datalake, chunk_size)
    load_moras(datalake, chunk_size)
    load_pagos(datalake, chunk_size)
    load_canales(datalake, chunk_size)

    logger.info("All debit tables loaded successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Load debit CSVs into PostgreSQL.")
    parser.add_argument("--datalake-path", default="./datalake")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--truncate", action="store_true", help="Truncate tables before loading")
    args = parser.parse_args()

    run(
        datalake_path=args.datalake_path,
        chunk_size=args.chunk_size,
        truncate=args.truncate,
    )
