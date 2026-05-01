"""
Feature selection supervisada para el modelo de débitos recurrentes — Bancolombia.

Pipeline en 3 pasos ajustados SOLO sobre el conjunto de entrenamiento:
  1. Filtro de varianza cero y sparsity extrema (no supervisado)
  2. ANOVA F-test: retiene el top-percentil por F-score univariado
  3. ElasticNet: LogisticRegression con penalty L1+L2; elimina features con coef = 0

Anti-leakage: los pasos 2 y 3 hacen fit únicamente sobre train.parquet.
El feature_cols.json resultante se aplica a test y OOT sin ningún refiteo.

Pipeline step (standalone):
    python -m src.dataset.feature_selection \
        --train-path data/artifacts/train.parquet \
        --output-dir data/artifacts
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]
TARGET_COL = "var_rta"

# ─── Parámetros por defecto ───────────────────────────────────────────────────
_VAR_THRESHOLD    = 0.0   # varianza exactamente cero → eliminar
_SPARSITY_LIMIT   = 0.99  # >= 99% ceros → eliminar
_ANOVA_PERCENTILE = 80    # top 80% por F-score ANOVA
_EN_L1_RATIO      = 0.7   # l1_ratio: más hacia L1 para mayor sparsity de coef
_EN_C             = 0.1   # inverso de regularización: 0.1 = agresivo
_EN_MAX_ITER      = 3000
_RANDOM_STATE     = 42
_CHUNK            = 500   # cols por chunk para el filtro de varianza (~90 MB / chunk)


# ─── Paso 1: varianza / sparsity ──────────────────────────────────────────────

def filter_low_variance(
    df_train: pd.DataFrame,
    candidates: list[str],
    var_threshold: float = _VAR_THRESHOLD,
    sparsity_limit: float = _SPARSITY_LIMIT,
) -> list[str]:
    """Eliminar features con varianza cero o sparsity extrema.

    Parameters
    ----------
    df_train : pd.DataFrame
        Partición de entrenamiento.
    candidates : list[str]
        Nombres de features candidatas (sin JOIN_KEYS ni TARGET_COL).
    var_threshold : float
        Umbral mínimo de varianza; features con var <= threshold se eliminan.
    sparsity_limit : float
        Fracción máxima de ceros; features con frac >= limit se eliminan.

    Returns
    -------
    list[str]
        Features que superan ambos filtros.

    Notes
    -----
    Procesamiento en chunks de _CHUNK columnas para mantener el pico de
    memoria por debajo de ~100 MB por iteración con float32.
    """
    keep: list[str] = []
    removed_var = removed_sparse = 0

    for i in range(0, len(candidates), _CHUNK):
        chunk = candidates[i : i + _CHUNK]
        arr = df_train[chunk].to_numpy(dtype=np.float32, na_value=0.0)
        var_vals = np.var(arr, axis=0)
        sparse_vals = (arr == 0).mean(axis=0)
        del arr

        for col, v, s in zip(chunk, var_vals, sparse_vals):
            if v <= var_threshold:
                removed_var += 1
            elif s >= sparsity_limit:
                removed_sparse += 1
            else:
                keep.append(col)

    logger.info(
        "Paso 1 — varianza/sparsity: eliminadas %d (var=0: %d | sparse>=%.0f%%: %d) → %d candidatas",
        removed_var + removed_sparse,
        removed_var,
        sparsity_limit * 100,
        removed_sparse,
        len(keep),
    )
    return keep


# ─── Paso 2: ANOVA F-test ─────────────────────────────────────────────────────

def select_anova(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: list[str],
    percentile: int = _ANOVA_PERCENTILE,
) -> tuple[list[str], np.ndarray]:
    """Seleccionar features por ANOVA F-test univariado.

    Parameters
    ----------
    X_train : np.ndarray
        Matriz de features del conjunto de entrenamiento (n_samples, n_features).
    y_train : np.ndarray
        Vector target binario.
    candidates : list[str]
        Nombres de features en el mismo orden que las columnas de X_train.
    percentile : int
        Porcentaje de features a retener ordenadas por F-score descendente.

    Returns
    -------
    tuple[list[str], np.ndarray]
        (features_seleccionadas, X_filtrado) — X_filtrado ya tiene solo las cols retenidas.

    Notes
    -----
    Fit exclusivo sobre train: SelectPercentile no ve labels de test/OOT.
    """
    selector = SelectPercentile(f_classif, percentile=percentile)
    X_sel = selector.fit_transform(X_train, y_train)
    mask = selector.get_support()
    selected = [c for c, m in zip(candidates, mask) if m]

    logger.info(
        "Paso 2 — ANOVA F-test (top %d%%): %d → %d features",
        percentile, len(candidates), len(selected),
    )
    return selected, X_sel


# ─── Paso 3: ElasticNet ───────────────────────────────────────────────────────

def select_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: list[str],
    l1_ratio: float = _EN_L1_RATIO,
    C: float = _EN_C,
    max_iter: int = _EN_MAX_ITER,
    random_state: int = _RANDOM_STATE,
) -> list[str]:
    """Seleccionar features via ElasticNet: retiene aquellas con coef != 0.

    Parameters
    ----------
    X_train : np.ndarray
        Matriz de features post-ANOVA (ya filtrada y sin escalar).
    y_train : np.ndarray
        Vector target binario.
    candidates : list[str]
        Nombres de features en el mismo orden que las columnas de X_train.
    l1_ratio : float
        Mezcla L1/L2: 1.0 = L1 puro (mayor sparsity), 0.0 = L2 puro (Ridge).
    C : float
        Inverso de la fuerza de regularización; valores menores son más agresivos.
    max_iter : int
        Iteraciones máximas del solver SAGA.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    list[str]
        Nombres de features con coeficiente absoluto > 0 tras el ajuste.

    Notes
    -----
    StandardScaler garantiza que los coeficientes sean comparables entre features
    con escalas distintas (ej. monto en pesos vs. ratio 0–1).
    Fit exclusivo sobre train: scaler y lr no ven datos de test/OOT.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    lr.fit(X_scaled, y_train)

    coef = np.abs(lr.coef_[0])
    selected = [c for c, v in zip(candidates, coef) if v > 0.0]

    logger.info(
        "Paso 3 — ElasticNet (l1_ratio=%.1f, C=%.2f): %d → %d features (coef > 0)",
        l1_ratio, C, len(candidates), len(selected),
    )
    return selected


# ─── Pipeline completo ────────────────────────────────────────────────────────

def run_feature_selection(
    train_path: Path,
    output_dir: Path,
    anova_percentile: int = _ANOVA_PERCENTILE,
    elasticnet_l1: float = _EN_L1_RATIO,
    elasticnet_C: float = _EN_C,
) -> list[str]:
    """Ejecutar el pipeline completo de selección sobre el conjunto de entrenamiento.

    Pasos:
      1. Filtro de varianza cero y sparsity extrema (no supervisado).
      2. ANOVA F-test: top-percentil por F-score univariado.
      3. ElasticNet (LogisticRegression L1+L2): coef != 0.

    Guarda el resultado en ``output_dir/feature_cols.json``.

    Parameters
    ----------
    train_path : Path
        Ruta al parquet de entrenamiento generado por feature_engineering.
    output_dir : Path
        Directorio donde se escribe feature_cols.json.
    anova_percentile : int
        Percentil top para ANOVA F-test.
    elasticnet_l1 : float
        l1_ratio para ElasticNet.
    elasticnet_C : float
        C (inverso de regularización) para ElasticNet.

    Returns
    -------
    list[str]
        Lista final de features seleccionadas.
    """
    logger.info("Cargando train desde %s", train_path)
    df_train = pd.read_parquet(train_path)
    logger.info(
        "Train cargado: %d filas x %d cols",
        len(df_train), len(df_train.columns),
    )

    exclude = set(JOIN_KEYS) | {TARGET_COL}
    candidates = [c for c in df_train.columns if c not in exclude]
    y = df_train[TARGET_COL].astype(int).to_numpy()

    logger.info("Candidatas iniciales: %d features", len(candidates))

    # Paso 1: varianza / sparsity (ajustado sobre train)
    candidates = filter_low_variance(df_train, candidates)

    # Paso 2: ANOVA F-test (ajustado sobre train)
    X = df_train[candidates].fillna(0.0).to_numpy(dtype=np.float32)
    candidates, X = select_anova(X, y, candidates, percentile=anova_percentile)

    # Paso 3: ElasticNet (ajustado sobre train)
    feature_cols = select_elasticnet(
        X, y, candidates, l1_ratio=elasticnet_l1, C=elasticnet_C,
    )

    out = output_dir / "feature_cols.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(feature_cols, fh, indent=2)

    logger.info(
        "Selección completada: %d → %d features finales | guardado en %s",
        len([c for c in df_train.columns if c not in exclude]),
        len(feature_cols),
        out,
    )
    return feature_cols


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Selección supervisada de features para débitos recurrentes."
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Ruta a train.parquet generado por feature_engineering",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se escribe feature_cols.json",
    )
    parser.add_argument(
        "--anova-percentile",
        type=int,
        default=_ANOVA_PERCENTILE,
        help=f"Top %% de features por ANOVA F-score (default: {_ANOVA_PERCENTILE})",
    )
    parser.add_argument(
        "--elasticnet-l1",
        type=float,
        default=_EN_L1_RATIO,
        help=f"l1_ratio para ElasticNet (default: {_EN_L1_RATIO})",
    )
    parser.add_argument(
        "--elasticnet-c",
        type=float,
        default=_EN_C,
        help=f"C — inverso de regularización — para ElasticNet (default: {_EN_C})",
    )
    args = parser.parse_args()

    run_feature_selection(
        train_path=Path(args.train_path),
        output_dir=Path(args.output_dir),
        anova_percentile=args.anova_percentile,
        elasticnet_l1=args.elasticnet_l1,
        elasticnet_C=args.elasticnet_c,
    )
