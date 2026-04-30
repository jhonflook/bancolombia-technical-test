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
import json
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
    trx_mnt = df.get("trx_mnt_total", pd.Series(0.0, index=df.index)).fillna(0.0)

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
    df = df.copy()
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
    """
    exclude = set(JOIN_KEYS) | {TARGET_COL}
    candidates = [c for c in df.columns if c not in exclude]

    # Varianza cero — columnas constantes
    variances = df[candidates].var(numeric_only=True)
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logger.info("Eliminando %d columnas con varianza cero", len(zero_var))
    candidates = [c for c in candidates if c not in zero_var]

    # Sparsity > 99% — columnas casi siempre en cero (mayoría de canales individuales)
    sparsity = (df[candidates] == 0).mean()
    too_sparse = sparsity[sparsity > 0.99].index.tolist()
    if too_sparse:
        logger.info("Eliminando %d columnas con >99%% ceros", len(too_sparse))
    candidates = [c for c in candidates if c not in too_sparse]

    logger.info("Features seleccionadas: %d", len(candidates))
    return candidates


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build debit feature matrix: apply engineering + temporal split."
    )
    parser.add_argument("--input", required=True, help="Path to analytical_model.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to write split parquets")
    args = parser.parse_args()

    from src.dataset.data_preparation import split_train_test_oot

    logger.info("Cargando modelo analítico desde %s", args.input)
    df = pd.read_parquet(args.input)

    logger.info("Construyendo features derivadas...")
    df = build_debit_features(df)

    feature_cols = select_feature_columns(df)
    logger.info("Features seleccionadas: %d columnas", len(feature_cols))

    df_train, df_test, df_oot = split_train_test_oot(df)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    save_cols = list(dict.fromkeys(JOIN_KEYS + [TARGET_COL] + feature_cols))

    df_train[save_cols].to_parquet(output / "train.parquet", index=False)
    df_test[save_cols].to_parquet(output / "test.parquet", index=False)
    df_oot[save_cols].to_parquet(output / "oot.parquet", index=False)

    with open(output / "feature_cols.json", "w") as fh:
        json.dump(feature_cols, fh, indent=2)

    logger.info(
        "Splits guardados: train=%d, test=%d, oot=%d | features=%d",
        len(df_train), len(df_test), len(df_oot), len(feature_cols),
    )
