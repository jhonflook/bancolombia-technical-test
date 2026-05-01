"""Feature engineering para el modelo de débitos recurrentes — Bancolombia.

Construye features derivadas que capturan las DOS condiciones del target:
  1. Exclusividad de canal (solo débito, sin otros canales)
  2. Recurrencia >= 40% de los pagos por débito

Pipeline step 3 (standalone):
    python -m src.dataset.feature_engineering \
        --input /app/data/artifacts/analytical_model.parquet \
        --output-dir /app/data/artifacts

Leakage (S9): las columnas avg/min/max/stddev_pago_debito_*m son agregaciones
históricas de ventanas ANTERIORES a f_analisis. No hay leakage temporal: los
valores representan el comportamiento pasado, no el evento que genera var_rta.
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]
TARGET_COL = "var_rta"
WINDOWS = ["3m", "6m", "9m", "12m"]
MORA_WINDOWS = ["3m", "6m", "9m"]  # S6: no existe ventana 12m en moras


# ─── Features derivadas ───────────────────────────────────────────────────────

def _pagos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construir features de exclusividad y proporción de canal débito.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico con columnas avg_pago_* por ventana.

    Returns
    -------
    pd.DataFrame
        df con columnas adicionales de exclusividad y proporción.

    Notes
    -----
    S9: avg_pago_debito_Xm es historial previo; incluir es válido (no leakage).
    S8: estas features son la señal más fuerte para el target binario.
    """
    idx = df.index
    for w in WINDOWS:
        deb = df.get(f"avg_pago_debito_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        fis = df.get(f"avg_pago_fisico_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        vir = df.get(f"avg_pago_virtual_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        otr = df.get(f"avg_pago_otros_{w}", pd.Series(0.0, index=idx)).fillna(0.0)

        total = deb + fis + vir + otr

        # Proporción de débito sobre total pagado en la ventana (S9 principal feature)
        df[f"prop_debito_{w}"] = np.where(total > 0, deb / total, 0.0)

        # Flag: solo canal débito activo en la ventana (condición exclusividad)
        df[f"canal_unico_{w}"] = (
            (fis == 0) & (vir == 0) & (otr == 0) & (deb > 0)
        ).astype(np.int8)

        # Flag: alguna actividad de débito en la ventana
        df[f"has_debito_{w}"] = (deb > 0).astype(np.int8)

        # Señal conjunta: exclusivo + recurrencia >= 40% (proxy del target)
        df[f"debito_exclusivo_40pct_{w}"] = (
            (df[f"canal_unico_{w}"] == 1) & (df[f"prop_debito_{w}"] >= 0.40)
        ).astype(np.int8)

        # Variabilidad del pago débito (estabilidad de canal)
        std_deb = df.get(f"stddev_pago_debito_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        df[f"cv_debito_{w}"] = std_deb / (deb + 1.0)

    return df


def _gestiones_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construir ratios de gestiones de cobranza como señales de comportamiento.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    idx = df.index
    for w in WINDOWS:
        gest = df.get(f"avg_cant_gestiones_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        acuerdo = df.get(f"avg_cant_acuerdo_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        rpc = df.get(f"avg_cant_rpc_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        promesas = df.get(f"avg_promesas_cumplidas_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        rank = df.get(f"avg_maximo_rank_{w}", pd.Series(0.0, index=idx)).fillna(0.0)

        # Eficiencia de acuerdos sobre total gestiones
        df[f"ratio_acuerdos_{w}"] = acuerdo / (gest + 1.0)
        # Tasa de RPC (right-party contact) sobre gestiones
        df[f"tasa_rpc_{w}"] = rpc / (gest + 1.0)
        # Cumplimiento de promesas sobre RPC
        df[f"promesas_pct_{w}"] = promesas / (rpc + 1.0)
        # Intensidad de gestión (rank máximo alcanzado)
        df[f"rank_norm_{w}"] = rank / (rank.max() + 1.0)

    return df


def _moras_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construir features de estabilidad de mora.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    S6: moras no tiene ventana 12m; solo 3m, 6m, 9m.
    """
    idx = df.index
    for w in MORA_WINDOWS:
        avg = df.get(f"moras_avg_mora_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        std = df.get(f"moras_stddev_mora_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        mn = df.get(f"moras_min_mora_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        mx = df.get(f"moras_max_mora_{w}", pd.Series(0.0, index=idx)).fillna(0.0)

        # Coeficiente de variación: mayor estabilidad => menor CV => más probable débito recurrente
        df[f"cv_mora_{w}"] = std / (avg + 1.0)

        # Rango de mora: amplitud de oscilación
        df[f"rango_mora_{w}"] = mx - mn

    return df


def _excedentes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construir indicadores de sobrepago sobre cuota.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    idx = df.index
    for w in WINDOWS:
        exc = df.get(f"avg_excedente_pago_{w}", pd.Series(0.0, index=idx)).fillna(0.0)
        pct = df.get(f"avg_porc_pago_{w}", pd.Series(0.0, index=idx)).fillna(0.0)

        df[f"has_excedente_{w}"] = (exc > 0).astype(np.int8)
        # Monto de sobrepago relativo: >1 => paga más que la cuota => buen pagador
        df[f"sobrepago_{w}"] = np.clip(pct - 1.0, 0.0, None)

    return df


def _canales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construir feature de actividad transaccional (cobertura 20.7%).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    S4: ausencia en canales = sin actividad => 0 (ya imputado upstream).
    S10: las ~4252 features individuales de canales se usan a nivel
         de resumen aquí; selección fina se aplica en select_feature_columns().
    """
    trx_cnt = df.get("trx_cnt_total", pd.Series(0.0, index=df.index)).fillna(0.0)

    # Flag: tiene actividad transaccional registrada en canales
    df["has_trx_canales"] = (trx_cnt > 0).astype(np.int8)

    # Monto total en unidades SMMLV (escala normalizada)
    df["trx_mnt_smmlv_log1p"] = np.log1p(
        df.get("trx_mnt_total_smmlv", pd.Series(0.0, index=df.index)).fillna(0.0)
    )

    return df


# ─── Selección de features ────────────────────────────────────────────────────

def build_debit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplicar todas las transformaciones de feature engineering al modelo analítico.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico producido por data_preparation.build_analytical_model().

    Returns
    -------
    pd.DataFrame
        df con columnas derivadas añadidas (no elimina las originales).
    """
    # No df.copy(): las sub-funciones solo añaden columnas; no se duplica ~1.7 GB
    df = _pagos_features(df)
    df = _gestiones_features(df)
    df = _moras_features(df)
    df = _excedentes_features(df)
    df = _canales_features(df)

    derived = [c for c in df.columns if any(
        c.startswith(p) for p in (
            "prop_debito_", "canal_unico_", "has_debito_", "debito_exclusivo_",
            "cv_debito_", "ratio_acuerdos_", "tasa_rpc_", "promesas_pct_",
            "rank_norm_", "cv_mora_", "rango_mora_", "has_excedente_",
            "sobrepago_", "has_trx_canales", "trx_mnt_smmlv_log1p",
        )
    )]
    logger.info("Feature engineering: %d features derivadas construidas", len(derived))
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Seleccionar columnas de features para el modelo de clasificación.

    Excluye: claves de join, target, varianza cero, sparsity > 99%.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con todas las features (base + derivadas).

    Returns
    -------
    list[str]
        Lista de nombres de columnas de features seleccionadas.

    Notes
    -----
    S10: canales tiene ~4252 features; el filtro de varianza y sparsity
         elimina la mayoría de las columnas ruidosas sin información.
    Implementación: pasada única por chunk usando numpy float32 para
    calcular varianza y sparsity simultáneamente. Evita crear dos
    DataFrames intermedios por chunk y reduce el pico de memoria a
    ~200 MB por iteración frente a ~500 MB del enfoque anterior.
    """
    exclude = set(JOIN_KEYS) | {TARGET_COL}
    candidates = [c for c in df.columns if c not in exclude]

    # Pasada única: varianza y sparsity en el mismo bloque numpy
    _CHUNK = 1000  # 1000 cols × 46 736 filas × float32 ≈ 178 MB por chunk
    keep: list[str] = []
    removed_var = 0
    removed_sparse = 0

    for i in range(0, len(candidates), _CHUNK):
        chunk = candidates[i:i + _CHUNK]
        # to_numpy(float32) evita copiar el DataFrame completo como objeto pandas
        arr = df[chunk].to_numpy(dtype=np.float32, na_value=0.0)
        var_vals = np.var(arr, axis=0)
        sparse_vals = (arr == 0).mean(axis=0)
        del arr

        for col, v, s in zip(chunk, var_vals, sparse_vals):
            if v == 0.0:
                removed_var += 1
            elif s > 0.99:
                removed_sparse += 1
            else:
                keep.append(col)

        gc.collect()

    logger.info(
        "Eliminando %d varianza-cero + %d >99%% ceros → %d features seleccionadas",
        removed_var, removed_sparse, len(keep),
    )
    return keep


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retornar matriz X de features y vector y de target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con features y columna var_rta.
    feature_cols : list[str] or None
        Si es None, aplica auto-selección vía select_feature_columns().

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) — listas de numeric features y target binario.
    """
    if feature_cols is None:
        feature_cols = select_feature_columns(df)

    X = df[feature_cols].fillna(0.0)
    y = df[TARGET_COL].astype(int)

    return X, y


# ─── Entrypoint pipeline ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import pyarrow as pa
    import pyarrow.parquet as pq

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build debit feature matrix: apply engineering + temporal split."
    )
    parser.add_argument("--input", required=True, help="Path to analytical_model.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to write split parquets")
    args = parser.parse_args()

    from src.dataset.data_preparation import (
        TRAIN_END, TEST_START, TEST_END, OOT_START,
    )
    from src.dataset.feature_config import DEFAULT_FEATURE_CONFIG

    logger.info("Cargando modelo analítico desde %s", args.input)
    # Leer via pyarrow y castear float64 → float32 ANTES de convertir a pandas.
    # Sin este paso: pico Arrow(float64) + pandas(float64) ≈ 2.2 GB → OOM.
    # Con cast en Arrow: pico ≈ 1.65 GB → encaja en los 2.9 GB disponibles.
    _tbl = pq.read_table(args.input)
    _new_schema = pa.schema([
        f.with_type(pa.float32()) if f.type == pa.float64() else f
        for f in _tbl.schema
    ])
    _tbl = _tbl.cast(_new_schema)
    df = _tbl.to_pandas()
    del _tbl
    gc.collect()
    logger.info(
        "Parquet cargado como float32: %d filas x %d cols | %.2f GB",
        len(df), len(df.columns), df.memory_usage(deep=False).sum() / 1024**3,
    )

    logger.info("Construyendo features derivadas...")
    df = build_debit_features(df)

    # Guardar todas las features candidatas definidas en el config (base + derivadas).
    # La selección supervisada (varianza → ANOVA → ElasticNet) se hace en el paso
    # siguiente: src.dataset.feature_selection, que ajusta SOLO sobre train.
    cfg_features = [c for c in DEFAULT_FEATURE_CONFIG.all_features() if c in df.columns]
    save_cols = list(dict.fromkeys(JOIN_KEYS + [TARGET_COL] + cfg_features))

    # Reducir df a save_cols antes del split para minimizar memoria en los slices
    df = df[save_cols]
    gc.collect()
    logger.info(
        "Features candidatas a guardar: %d columnas (selección supervisada pendiente)",
        len(cfg_features),
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Splits secuenciales: nunca más de un split + df en memoria al mismo tiempo
    f = df["f_analisis"]
    splits = {
        "train": df[f <= TRAIN_END],
        "test":  df[(f >= TEST_START) & (f <= TEST_END)],
        "oot":   df[f >= OOT_START],
    }
    counts: dict[str, int] = {}
    for name, part in splits.items():
        counts[name] = len(part)
        part.to_parquet(output / f"{name}.parquet", index=False)
        logger.info("Guardado %s.parquet: %d filas", name, len(part))
        gc.collect()

    del df, splits, f
    gc.collect()

    logger.info(
        "Splits guardados: train=%d, test=%d, oot=%d | %d features candidatas "
        "→ ejecutar feature_selection para generar feature_cols.json",
        counts["train"], counts["test"], counts["oot"], len(cfg_features),
    )
