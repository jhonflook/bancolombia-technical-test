"""Modelo de datos analítico para débitos recurrentes — Bancolombia.

Pipeline step 2 (standalone):
    python -m src.dataset.data_preparation \
        --datalake-path /app/datalake \
        --output-dir /app/data/artifacts


Granularidad: cliente x obligación x periodo (num_doc, obl17, f_analisis).
Pipeline: carga -> join -> imputación -> validación -> split temporal.

Supuestos activos (referenciados en el código):
  S1  PK = (num_doc, obl17, f_analisis).
  S2  num_doc y obl17 son hashes anonimizados; solo se usan como llaves.
  S4  Canales cubre el 20.7% de las obligaciones; ausencia = sin actividad -> 0.
  S5  Nulos en excedentes crecen con la ventana; obligaciones sin historial -> 0.
  S7  Split temporal: Train 8 periodos / Test 5 periodos / OOT 4 periodos.
  S8  Desbalance 78.8% / 21.2%; mantener distribución natural por partición.
  S9  pago_debito_* en pagos es señal directa de var_rta -> riesgo de leakage.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constantes públicas (consumidas por DAGs y módulos downstream) ──────────

JOIN_KEYS: list[str] = ["num_doc", "obl17", "f_analisis"]
TARGET_COL: str = "var_rta"

# Límites del split temporal — Supuesto S7
TRAIN_END: pd.Timestamp = pd.Timestamp("2025-02-01")   # Train:  2024-07-01 – 2025-02-01
TEST_START: pd.Timestamp = pd.Timestamp("2025-03-01")  # Test:   2025-03-01 – 2025-07-01
TEST_END: pd.Timestamp = pd.Timestamp("2025-07-01")
OOT_START: pd.Timestamp = pd.Timestamp("2025-08-01")   # OOT:    2025-08-01 – 2025-11-01

# ─── Constantes privadas ─────────────────────────────────────────────────────

_CSV_FILES: dict[str, str] = {
    "clientes":   "clientes_seleccionados_prueba.csv",
    "canales":    "debitos_dev_cruce_canales_poblacion_sel_vars_prueba_tecnica_1.csv",
    "excedentes": "debitos_dev_cruce_excedente_poblacion_prueba_tecnica.csv",
    "gestiones":  "debitos_dev_cruce_gestiones_poblacion_prueba_tecnica.csv",
    "moras":      "debitos_dev_cruce_moras_poblacion_prueba_tecnica.csv",
    "pagos":      "debitos_dev_cruce_tanque_poblacion_prueba_tecnica.csv",
}


# ─── Carga ───────────────────────────────────────────────────────────────────


def load_raw_sources(datalake_path: Path | str = "datalake") -> dict[str, pd.DataFrame]:
    """Cargar los 6 CSVs fuente desde el datalake.

    Parameters
    ----------
    datalake_path : Path or str
        Directorio que contiene los archivos CSV del examen.

    Returns
    -------
    dict[str, pd.DataFrame]
        Claves: clientes, canales, excedentes, gestiones, moras, pagos.
        f_analisis se parsea como datetime; columnas numéricas se castean a float.

    Notes
    -----
    S1: la clave primaria compuesta se mantiene como string hasta el join.
    S2: num_doc y obl17 son hashes; no se infiere ningún atributo de ellos.
    """
    base = Path(datalake_path)
    sources: dict[str, pd.DataFrame] = {}

    for alias, filename in _CSV_FILES.items():
        df = pd.read_csv(base / filename, sep=None, engine="python", dtype=str)
        df["f_analisis"] = pd.to_datetime(df["f_analisis"])

        non_key = [c for c in df.columns if c not in JOIN_KEYS]
        for col in non_key:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        sources[alias] = df
        logger.info("Cargado %-12s %6d filas x %d cols", alias, len(df), len(df.columns))

    return sources


# ─── Construcción del modelo analítico ───────────────────────────────────────


def build_analytical_model(datalake_path: Path | str = "datalake") -> pd.DataFrame:
    """Construir el modelo analítico unificado mediante LEFT JOINs.

    Usa clientes como tabla base (contiene var_rta) y agrega las features
    de las otras cinco fuentes sobre la clave (num_doc, obl17, f_analisis).

    Parameters
    ----------
    datalake_path : Path or str
        Directorio del datalake.

    Returns
    -------
    pd.DataFrame
        Modelo analítico: ~46 736 filas x (4 llaves + features).
        Una fila por (cliente, obligación, periodo).

    Notes
    -----
    S1  — PK = (num_doc, obl17, f_analisis).
    S4  — canales cubre solo el 20.7% de las obligaciones; las filas
           ausentes representan cero actividad transaccional -> fill(0).
    S5  — nulos en excedentes se rellenan con 0 (historial insuficiente).
    S9  — pago_debito_* es señal directa de var_rta; evaluar leakage
           antes de entrenar.
    """
    sources = load_raw_sources(datalake_path)
    base = sources["clientes"].copy()

    # Fuentes con cobertura completa (45 731 obligaciones únicas c/u)
    for alias in ("excedentes", "gestiones", "moras", "pagos"):
        feat = sources[alias].drop_duplicates(subset=JOIN_KEYS)
        base = base.merge(feat, on=JOIN_KEYS, how="left")
        logger.info("Tras join %-12s %d filas", alias, len(base))

    # canales: cobertura parcial ~20.7% — S4: ausencia = sin actividad -> 0
    canales = sources["canales"].drop_duplicates(subset=JOIN_KEYS)
    canales_feat_cols = [c for c in canales.columns if c not in JOIN_KEYS]
    base = base.merge(canales, on=JOIN_KEYS, how="left")
    base[canales_feat_cols] = base[canales_feat_cols].fillna(0)
    logger.info("Tras join %-12s %d filas", "canales", len(base))

    base = _impute_excedentes(base)

    dist = base[TARGET_COL].value_counts().to_dict() if TARGET_COL in base.columns else {}
    logger.info(
        "Modelo analítico listo: %d filas x %d cols | dist target: %s",
        len(base), len(base.columns), dist,
    )
    return base


def _impute_excedentes(df: pd.DataFrame) -> pd.DataFrame:
    """Imputar con 0 los nulos de excedentes y porcentaje de pago.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo tras el join de todas las fuentes.

    Returns
    -------
    pd.DataFrame
        DataFrame con nulos de excedentes resueltos.

    Notes
    -----
    S5: los nulos crecen con la ventana temporal (3m -> 12m) porque las
    obligaciones jóvenes no tienen historial suficiente. Cero es el valor
    correcto: sin excedente registrado equivale a excedente = 0.
    """
    exc_cols = [
        c for c in df.columns
        if "excedente_pago" in c or "porc_pago" in c
    ]
    if exc_cols:
        df = df.copy()
        df[exc_cols] = df[exc_cols].fillna(0)
        logger.debug("Imputados %d cols de excedentes con 0 (S5)", len(exc_cols))
    return df


# ─── Validación ──────────────────────────────────────────────────────────────


def validate_join_integrity(df: pd.DataFrame) -> dict:
    """Validar unicidad de clave y calidad del modelo analítico.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico producido por build_analytical_model.

    Returns
    -------
    dict
        Reporte con las siguientes claves:

        total_filas : int
        claves_unicas : int
        claves_duplicadas : int
        periodos : list[str]
            Valores de f_analisis presentes (formato YYYY-MM-DD).
        distribucion_target : dict
            {0: n, 1: n} de var_rta.
        cobertura_canales_pct : float
            Porcentaje de filas con al menos un valor trx_ distinto de 0.
        nulos_restantes : dict
            Columnas con nulos y su conteo tras la imputación.
        ok : bool
            True si no hay duplicados ni nulos residuales.
    """
    report: dict = {}

    report["total_filas"] = len(df)
    report["claves_unicas"] = df[JOIN_KEYS].drop_duplicates().shape[0]
    report["claves_duplicadas"] = report["total_filas"] - report["claves_unicas"]
    report["periodos"] = sorted(df["f_analisis"].dt.strftime("%Y-%m-%d").unique().tolist())

    if TARGET_COL in df.columns:
        report["distribucion_target"] = df[TARGET_COL].value_counts().to_dict()

    trx_cols = [c for c in df.columns if c.startswith("trx_")]
    if trx_cols:
        con_actividad = (df[trx_cols] > 0).any(axis=1)
        report["cobertura_canales_pct"] = round(con_actividad.mean() * 100, 2)

    nulos = df.isnull().sum()
    report["nulos_restantes"] = nulos[nulos > 0].to_dict()

    report["ok"] = (
        report["claves_duplicadas"] == 0
        and len(report["nulos_restantes"]) == 0
    )

    if report["ok"]:
        logger.info("Validación de integridad superada.")
    else:
        logger.warning("Problemas en validación: %s", report)

    return report


# ─── Split temporal ───────────────────────────────────────────────────────────


def split_train_test_oot(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dividir el modelo analítico en Train, Test y OOT por f_analisis.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico con f_analisis como columna datetime.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_test, df_oot) — particiones sin solapamiento temporal.

    Notes
    -----
    S7 — límites del split:
      Train : 2024-07-01 – 2025-02-01  (8 periodos)
      Test  : 2025-03-01 – 2025-07-01  (5 periodos)
      OOT   : 2025-08-01 – 2025-11-01  (4 periodos)

    S8 — el desbalance de clases (78.8% / 21.2%) se preserva de forma
    natural al respetar la cronología. Aplicar estratificación dentro de
    cada partición al momento de entrenar, no aquí.
    """
    f = df["f_analisis"]

    df_train = df[f <= TRAIN_END].copy()
    df_test  = df[(f >= TEST_START) & (f <= TEST_END)].copy()
    df_oot   = df[f >= OOT_START].copy()

    for nombre, part in (("train", df_train), ("test", df_test), ("oot", df_oot)):
        periodos = sorted(part["f_analisis"].dt.strftime("%Y-%m-%d").unique().tolist())
        dist = (
            part[TARGET_COL].value_counts(normalize=True).round(3).to_dict()
            if TARGET_COL in part.columns else {}
        )
        logger.info(
            "Split %-6s %5d filas | %d periodos (%s … %s) | dist: %s",
            nombre, len(part), len(periodos),
            periodos[0] if periodos else "-",
            periodos[-1] if periodos else "-",
            dist,
        )

    return df_train, df_test, df_oot


# ─── Trazabilidad de features ─────────────────────────────────────────────────


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Retornar columnas de features agrupadas por tabla fuente.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico unificado.

    Returns
    -------
    dict[str, list[str]]
        Claves: moras, excedentes, gestiones, pagos, canales.
        Valores: listas de columnas pertenecientes a cada fuente.

    Notes
    -----
    S9 — las columnas de pagos que contengan pago_debito son señal
    directa de var_rta; evaluar su exclusión antes de entrenar.
    """
    metadata = set(JOIN_KEYS) | {TARGET_COL}

    groups: dict[str, list[str]] = {
        "moras": [
            c for c in df.columns if c.startswith("moras_")
        ],
        "excedentes": [
            c for c in df.columns
            if "excedente_pago" in c or "porc_pago" in c
        ],
        "gestiones": [
            c for c in df.columns
            if any(k in c for k in (
                "cant_gestiones", "cant_rpc", "cant_acuerdo",
                "promesas_cumplidas", "maximo_rank",
            ))
        ],
        "pagos": [
            c for c in df.columns
            if any(k in c for k in (
                "pago_debito", "pago_fisico", "pago_virtual", "pago_otros",
            ))
        ],
        # canales: columnas punto-en-tiempo (trx_*) y agregadas (avg/min/max/stddev/sum_trx_*_ultN)
        "canales": [
            c for c in df.columns if "trx_" in c
        ],
    }

    asignadas: set[str] = set()
    for fuente, cols in groups.items():
        solapamiento = asignadas & set(cols)
        if solapamiento:
            logger.warning("Columnas en múltiples grupos: %s", solapamiento)
        asignadas.update(cols)

    sin_grupo = set(df.columns) - metadata - asignadas
    if sin_grupo:
        logger.debug("Columnas sin grupo asignado: %s", sin_grupo)

    return groups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build debit analytical model and save as parquet.")
    parser.add_argument("--datalake-path", default="./datalake")
    parser.add_argument("--output-dir", default="./data/artifacts")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = build_analytical_model(datalake_path=args.datalake_path)
    report = validate_join_integrity(df)

    if not report["ok"]:
        logger.warning("Integrity issues detected: %s", report)

    out_path = output / "analytical_model.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Analytical model saved to %s (%d rows x %d cols)", out_path, len(df), len(df.columns))
