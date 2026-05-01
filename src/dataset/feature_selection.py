"""Feature selection supervisada para el modelo de débitos recurrentes — Bancolombia.

Pipeline secuencial en 3 pasos, ajustados SOLO sobre el conjunto de entrenamiento:
  1. filter_low_variance: var<=0.01 (casi cero) o sparsity>=99% → eliminar (no supervisado).
  2. select_anova: SelectPercentile(f_classif, percentile=80) → top 80% por F-score.
  3. select_elasticnet: LogisticRegression(elasticnet, l1_ratio=0.7, C=0.1) → coef != 0.

La salida de cada paso es la entrada del siguiente (reducción progresiva).

Anti-leakage: todos los pasos supervisados ajustan SOLO sobre train.parquet.
El feature_cols.json resultante se aplica sin cambios a test y OOT.

Standalone:
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

JOIN_KEYS  = ["num_doc", "obl17", "f_analisis"]
TARGET_COL = "var_rta"

# ─── Parámetros por defecto ───────────────────────────────────────────────────
_VAR_THRESHOLD    = 0.01  # varianza <= umbral (cero o casi cero) → eliminar
_SPARSITY_LIMIT   = 0.99  # >= 99% ceros → eliminar
_ANOVA_PERCENTILE = 80    # top-80% por F-score ANOVA
_EN_L1_RATIO      = 0.7   # mezcla L1/L2 para ElasticNet (1.0 = L1 puro)
_EN_C             = 0.1   # inverso de regularización (menor = más agresivo)
_EN_MAX_ITER      = 3000
_RANDOM_STATE     = 42
_CHUNK            = 500   # columnas por bloque en el filtro de varianza


# ─── Paso 1: varianza / sparsity ──────────────────────────────────────────────

def filter_low_variance(
    df_train: pd.DataFrame,
    candidates: list[str],
    var_threshold: float = _VAR_THRESHOLD,
    sparsity_limit: float = _SPARSITY_LIMIT,
) -> list[str]:
    """Eliminar features con varianza cero o casi cero, o sparsity extrema.

    Parameters
    ----------
    df_train : pd.DataFrame
    candidates : list[str]
        Features candidatas (sin JOIN_KEYS ni TARGET_COL).
    var_threshold : float
        Features con varianza <= threshold se eliminan (cubre var=0 y casi cero).
        Default 0.01 — apropiado para features en escala ratio (0–1) y SMMLV.
    sparsity_limit : float
        Features con fracción de ceros >= limit se eliminan.

    Returns
    -------
    list[str]

    Notes
    -----
    Procesado en bloques de _CHUNK columnas para controlar uso de memoria
    (~90 MB por bloque con 22,000 filas en float32).
    """
    keep: list[str] = []
    removed_var = removed_sparse = 0

    for i in range(0, len(candidates), _CHUNK):
        chunk = candidates[i : i + _CHUNK]
        arr = df_train[chunk].fillna(0.0).to_numpy(dtype=np.float32)
        var_vals    = np.var(arr, axis=0)
        sparse_vals = (arr == 0).mean(axis=0)
        del arr

        for col, v, s in zip(chunk, var_vals, sparse_vals):
            if v <= var_threshold:
                removed_var += 1
            elif s >= sparsity_limit:  # noqa: SIM114
                removed_sparse += 1
            else:
                keep.append(col)

    logger.info(
        "Paso 1 — varianza/sparsity: eliminadas %d (var<=%.3f: %d | sparse>=%.0f%%: %d) → %d candidatas",
        removed_var + removed_sparse, var_threshold, removed_var,
        sparsity_limit * 100, removed_sparse, len(keep),
    )
    return keep


# ─── Paso 2: ANOVA F-test ─────────────────────────────────────────────────────

def select_anova(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[str],
    percentile: int = _ANOVA_PERCENTILE,
) -> list[str]:
    """Seleccionar top-percentil de features por ANOVA F-score univariado.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Valores de las features en `candidates` (ya filtradas por varianza).
    y : np.ndarray (n_samples,)
    candidates : list[str]
    percentile : int
        Porcentaje superior a retener (80 → top 80% por F-score).

    Returns
    -------
    list[str]
        Subset de `candidates` con F-score en el top `percentile`%.

    Notes
    -----
    sklearn implementa f_classif en C. Fit exclusivo sobre train.
    """
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit(X, y)
    selected = [c for c, m in zip(candidates, selector.get_support()) if m]
    logger.info(
        "Paso 2 — ANOVA (top %d%%): %d → %d features",
        percentile, len(candidates), len(selected),
    )
    return selected


# ─── Paso 3: ElasticNet ───────────────────────────────────────────────────────

def select_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[str],
    l1_ratio: float = _EN_L1_RATIO,
    C: float = _EN_C,
    max_iter: int = _EN_MAX_ITER,
    random_state: int = _RANDOM_STATE,
) -> list[str]:
    """Seleccionar features con coeficiente ElasticNet distinto de cero.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Valores de las features en `candidates` (ya filtradas por ANOVA).
    y : np.ndarray (n_samples,)
    candidates : list[str]
    l1_ratio : float
        Mezcla L1/L2 (1.0 = L1 puro, 0.0 = Ridge).
    C : float
        Inverso de la fuerza de regularización.
    max_iter : int
    random_state : int

    Returns
    -------
    list[str]
        Subset de `candidates` con coeficiente absoluto > 0.

    Notes
    -----
    StandardScaler garantiza comparabilidad entre features de escalas distintas
    (ej. monto en pesos vs. ratio 0–1). Fit exclusivo sobre train.
    """
    X_scaled = StandardScaler().fit_transform(X)
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
    lr.fit(X_scaled, y)

    coef     = np.abs(lr.coef_[0])
    selected = [c for c, v in zip(candidates, coef) if v > 0.0]
    logger.info(
        "Paso 3 — ElasticNet (l1=%.1f, C=%.2f): %d → %d features (coef > 0)",
        l1_ratio, C, len(candidates), len(selected),
    )
    return selected


# ─── Pipeline completo ────────────────────────────────────────────────────────

def run_feature_selection(
    train_path: Path,
    output_dir: Path,
    var_threshold: float    = _VAR_THRESHOLD,
    anova_percentile: int   = _ANOVA_PERCENTILE,
    elasticnet_l1: float    = _EN_L1_RATIO,
    elasticnet_C: float     = _EN_C,
) -> list[str]:
    """Ejecutar el pipeline secuencial de selección sobre el conjunto de entrenamiento.

    Flujo (secuencial — salida de cada paso es entrada del siguiente):
      1. filter_low_variance  → pre_candidates
      2. select_anova         → anova_candidates
      3. select_elasticnet    → final_features

    Salidas en output_dir:
      - feature_cols.json: lista final de features seleccionadas.

    Parameters
    ----------
    train_path : Path
    output_dir : Path
    anova_percentile : int
    elasticnet_l1 : float
    elasticnet_C : float

    Returns
    -------
    list[str]
        Features seleccionadas en el orden resultante del pipeline.
    """
    logger.info("Cargando train desde %s", train_path)
    df_train = pd.read_parquet(train_path)
    logger.info("Train: %d filas x %d cols", len(df_train), len(df_train.columns))

    exclude    = set(JOIN_KEYS) | {TARGET_COL}
    all_cands  = [c for c in df_train.columns if c not in exclude]
    y          = df_train[TARGET_COL].astype(int).to_numpy()

    # ── Paso 1: varianza / sparsity (no supervisado) ──────────────────────────
    step1 = filter_low_variance(df_train, all_cands, var_threshold=var_threshold)

    # Construir X con las candidatas post-paso 1
    X1 = df_train[step1].fillna(0.0).to_numpy(dtype=np.float32)
    logger.info(
        "Matriz X (paso 1→2): %d × %d (%.1f MB)",
        X1.shape[0], X1.shape[1], X1.nbytes / 1024**2,
    )

    # ── Paso 2: ANOVA (supervisado, fit solo en train) ─────────────────────────
    step2 = select_anova(X1, y, step1, percentile=anova_percentile)
    del X1

    # Reconstruir X con las candidatas post-paso 2
    X2 = df_train[step2].fillna(0.0).to_numpy(dtype=np.float32)
    logger.info(
        "Matriz X (paso 2→3): %d × %d (%.1f MB)",
        X2.shape[0], X2.shape[1], X2.nbytes / 1024**2,
    )

    # ── Paso 3: ElasticNet (supervisado, fit solo en train) ───────────────────
    final = select_elasticnet(X2, y, step2, l1_ratio=elasticnet_l1, C=elasticnet_C)
    del X2

    # ── Guardar feature_cols.json ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = output_dir / "feature_cols.json"
    with open(feat_path, "w") as fh:
        json.dump(final, fh, indent=2)

    logger.info(
        "Selección completada: %d candidatas → %d finales | %s",
        len(all_cands), len(final), feat_path,
    )
    return final


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Selección supervisada de features — débitos recurrentes."
    )
    parser.add_argument("--train-path",       required=True,
                        help="Ruta a train.parquet generado por feature_engineering")
    parser.add_argument("--output-dir",       required=True,
                        help="Directorio donde se escribe feature_cols.json")
    parser.add_argument("--var-threshold",    type=float, default=_VAR_THRESHOLD,
                        help=f"Varianza máxima para considerar una feature casi constante (default: {_VAR_THRESHOLD})")
    parser.add_argument("--anova-percentile", type=int,   default=_ANOVA_PERCENTILE,
                        help=f"Top %% ANOVA F-score (default: {_ANOVA_PERCENTILE})")
    parser.add_argument("--elasticnet-l1",    type=float, default=_EN_L1_RATIO,
                        help=f"l1_ratio ElasticNet (default: {_EN_L1_RATIO})")
    parser.add_argument("--elasticnet-c",     type=float, default=_EN_C,
                        help=f"C ElasticNet — inverso regularización (default: {_EN_C})")
    args = parser.parse_args()

    run_feature_selection(
        train_path       = Path(args.train_path),
        output_dir       = Path(args.output_dir),
        var_threshold    = args.var_threshold,
        anova_percentile = args.anova_percentile,
        elasticnet_l1    = args.elasticnet_l1,
        elasticnet_C     = args.elasticnet_c,
    )
