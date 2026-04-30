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
    """Load canales using a 2-pass streaming approach to avoid OOM.

    The source CSV has 4,258 columns. Loading it entirely + to_dict(orient="records")
    creates ~5 GB of Python objects, hanging low-memory machines.

    Pass 1 (lightweight): reads only KEY_COLS to identify duplicate keys
    (exact duplicates + S11 conflicting groups) without touching feature columns.
    Over-exclusion vs original logic: exact-duplicate rows are excluded rather
    than deduplicated (affects ~180 rows out of ~9,419 — negligible).

    Pass 2 (streaming): reads the CSV in small chunks, skips excluded keys,
    serializes feature columns to JSONB using .values.tolist() which converts
    numpy scalars to Python-native types required by json.dumps.

    S4: absent rows = no transactional activity; LEFT JOIN imputes 0 downstream.
    S11: key groups with conflicting values excluded (24 groups, 0.25% of universe).
    """
    csv_path = datalake / DATALAKE_FILES["canales"]
    # Cap chunk size: each row ~125 KB of JSON → 50 rows ≈ 6 MB per INSERT
    canales_chunk = min(chunk_size, 50)

    # --- Pass 1: detect all keys with any duplication ---
    logger.info("  canales: pre-scanning %d KEY_COLS for duplicate detection...", len(KEY_COLS))
    keys_scan = pd.read_csv(csv_path, usecols=KEY_COLS, low_memory=False)
    keys_scan["f_analisis"] = pd.to_datetime(keys_scan["f_analisis"]).dt.date
    dup_mask = keys_scan.duplicated(subset=KEY_COLS, keep=False)
    excluded_keys: set[tuple] = set()
    if dup_mask.any():
        excluded_keys = {
            tuple(row)
            for row in keys_scan.loc[dup_mask, KEY_COLS].drop_duplicates().itertuples(index=False)
        }
        logger.warning(
            "  canales: %d rows map to %d duplicate keys → excluded (S11 + exact-dup)",
            dup_mask.sum(), len(excluded_keys),
        )
    del keys_scan

    # --- Pass 2: stream full CSV, build JSONB per chunk ---
    logger.info("  canales: streaming CSV with chunk_size=%d", canales_chunk)
    loaded = 0
    for chunk in pd.read_csv(csv_path, chunksize=canales_chunk, low_memory=False):
        chunk["f_analisis"] = pd.to_datetime(chunk["f_analisis"]).dt.date

        if excluded_keys:
            key_tuples = list(zip(chunk["num_doc"], chunk["obl17"], chunk["f_analisis"]))
            chunk = chunk.loc[[k not in excluded_keys for k in key_tuples]]

        if chunk.empty:
            continue

        feature_cols = [c for c in chunk.columns if c not in KEY_COLS + CANALES_SUMMARY_COLS]
        rows = chunk[KEY_COLS + CANALES_SUMMARY_COLS].copy()

        # .values.tolist() converts numpy scalars → Python native (required by json.dumps)
        feat_values = chunk[feature_cols].fillna(0).values.tolist()
        rows["features"] = [json.dumps(dict(zip(feature_cols, v))) for v in feat_values]

        rows.to_sql("debit_canales", engine, if_exists="append", index=False, method="multi")
        loaded += len(rows)
        logger.info("  debit_canales: %d rows loaded so far", loaded)

    logger.info("  canales: %d total rows loaded", loaded)


def _truncate_debit_tables() -> None:
    tables = [
        "debit_canales", "debit_pagos", "debit_moras",
        "debit_gestiones", "debit_excedentes", "debit_clients",
    ]
    with engine.begin() as conn:
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    logger.info("All debit tables truncated.")


_LOADERS = {
    "clientes":   load_clientes,
    "excedentes": load_excedentes,
    "gestiones":  load_gestiones,
    "moras":      load_moras,
    "pagos":      load_pagos,
    "canales":    load_canales,
}


def run(
    datalake_path: str = "./datalake",
    chunk_size: int = 5000,
    truncate: bool = False,
    tables: list[str] | None = None,
) -> None:
    """Execute the load pipeline for debit source files.

    Parameters
    ----------
    datalake_path:
        Path to the directory containing the 6 CSV files.
    chunk_size:
        Number of rows per insert batch (canales is capped internally at 50).
    truncate:
        If True, truncate target tables before loading (idempotent re-run).
    tables:
        Subset of table aliases to load. None means all 6 tables.
    """
    datalake = Path(datalake_path)
    selected = tables if tables else list(_LOADERS)

    unknown = set(selected) - set(_LOADERS)
    if unknown:
        raise ValueError(f"Unknown table(s): {unknown}. Valid: {list(_LOADERS)}")

    if truncate:
        _truncate_debit_tables()

    for alias in selected:
        _LOADERS[alias](datalake, chunk_size)

    logger.info("Loaded tables: %s", selected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Load debit CSVs into PostgreSQL.")
    parser.add_argument("--datalake-path", default="./datalake")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--truncate", action="store_true", help="Truncate tables before loading")
    parser.add_argument(
        "--table",
        dest="tables",
        nargs="+",
        choices=list(_LOADERS),
        metavar="TABLE",
        help="Load only specific table(s): clientes excedentes gestiones moras pagos canales",
    )
    args = parser.parse_args()

    run(
        datalake_path=args.datalake_path,
        chunk_size=args.chunk_size,
        truncate=args.truncate,
        tables=args.tables,
    )
