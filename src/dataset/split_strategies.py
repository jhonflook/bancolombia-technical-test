"""Estrategias de split para el modelo de débitos recurrentes — Bancolombia.

Provee dos estrategias intercambiables para dividir el modelo analítico
en particiones Train / Test / OOT:

  temporal  (default, S7):
    Respeta la cronología de f_analisis.
    Train: 2024-07-01 – 2025-02-01  (8 periodos)
    Test:  2025-03-01 – 2025-07-01  (5 periodos)
    OOT:   2025-08-01 – 2025-11-01  (4 periodos)
    Evalúa estabilidad del modelo en cohortes nuevas (poblaciones distintas).

  random:
    Split aleatorio estratificado por var_rta.
    Mezcla periodos en las tres particiones manteniendo distribución de clases.
    Proporciones aproximadas al split temporal: train≈48% / test≈29% / oot≈23%.

    Diagnóstico principal: si AUC sigue en 1.0 con random split, el problema
    viene de separación perfecta en las features (independiente de la cronología),
    no de una ventaja informacional del split temporal.

Uso como módulo:
    from src.dataset.split_strategies import apply_split
    df_train, df_test, df_oot = apply_split(df, strategy="temporal")
    df_train, df_test, df_oot = apply_split(df, strategy="random", random_state=42)
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SplitStrategyName = Literal["temporal", "random"]

# Límites del split temporal — Supuesto S7
TRAIN_END: pd.Timestamp = pd.Timestamp("2025-02-01")
TEST_START: pd.Timestamp = pd.Timestamp("2025-03-01")
TEST_END: pd.Timestamp = pd.Timestamp("2025-07-01")
OOT_START: pd.Timestamp = pd.Timestamp("2025-08-01")

# Proporciones equivalentes al split temporal para el split aleatorio.
# Derivadas de los conteos reales: train=22291, test=13712, oot=10733, total=46736.
_DEFAULT_TRAIN_SIZE: float = 0.477
_DEFAULT_TEST_SIZE: float = 0.293  # fracción sobre el total (no sobre el resto)


# ─── Estrategia temporal ──────────────────────────────────────────────────────


def temporal_split(
    df: pd.DataFrame,
    target_col: str = "var_rta",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dividir el modelo analítico en Train/Test/OOT por f_analisis (S7).

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico con f_analisis como columna datetime.
    target_col : str
        Nombre de la columna target (solo para logging de distribución).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_test, df_oot) sin solapamiento temporal.

    Notes
    -----
    S7 — límites fijos definidos en este módulo (TRAIN_END, TEST_START, etc.).
    S8 — la distribución de clases se preserva naturalmente al respetar
         la cronología; no requiere estratificación explícita.
    S13 — el 97.9% de obligaciones aparece en un único periodo, por lo que
          el split segmenta cohortes de entrada, no evolución longitudinal.
    """
    f = df["f_analisis"]
    df_train = df[f <= TRAIN_END].copy()
    df_test  = df[(f >= TEST_START) & (f <= TEST_END)].copy()
    df_oot   = df[f >= OOT_START].copy()

    _log_partitions("temporal", df_train, df_test, df_oot, target_col)
    return df_train, df_test, df_oot


# ─── Estrategia aleatoria ─────────────────────────────────────────────────────


def random_split(
    df: pd.DataFrame,
    target_col: str = "var_rta",
    train_size: float = _DEFAULT_TRAIN_SIZE,
    test_size: float = _DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dividir el modelo analítico con split aleatorio estratificado.

    Mezcla todos los periodos disponibles en las tres particiones manteniendo
    la distribución de clases de var_rta. Las proporciones por defecto replican
    aproximadamente las del split temporal para hacer comparaciones directas.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico.
    target_col : str
        Columna de estratificación (var_rta).
    train_size : float
        Fracción del total para Train (default 0.477 ≈ split temporal).
    test_size : float
        Fracción del total para Test (default 0.293 ≈ split temporal).
        El resto va a OOT.
    random_state : int
        Semilla de reproducibilidad (default 42).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_test, df_oot) con proporciones equivalentes al split temporal.

    Notes
    -----
    El split se realiza en dos pasos para preservar estratificación:
      Paso 1: df → df_train + df_rest  (estratificado por target_col)
      Paso 2: df_rest → df_test + df_oot (estratificado por target_col)
    La fracción relativa de test sobre df_rest se recalcula para que la
    proporción global de test sea aproximadamente `test_size`.
    """
    if not 0 < train_size < 1:
        raise ValueError(f"train_size debe estar en (0, 1), recibido: {train_size}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size debe estar en (0, 1), recibido: {test_size}")
    if train_size + test_size >= 1.0:
        raise ValueError(
            f"train_size ({train_size}) + test_size ({test_size}) >= 1.0; "
            "no queda espacio para OOT"
        )

    df_train, df_rest = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Proporción de test relativa al resto para que la fracción global sea ~test_size
    rest_frac = 1.0 - train_size
    test_relative = test_size / rest_frac

    df_test, df_oot = train_test_split(
        df_rest,
        train_size=test_relative,
        stratify=df_rest[target_col],
        random_state=random_state,
    )

    _log_partitions("random", df_train, df_test, df_oot, target_col)
    return df_train, df_test, df_oot


# ─── Interfaz pública ─────────────────────────────────────────────────────────


def apply_split(
    df: pd.DataFrame,
    strategy: SplitStrategyName,
    target_col: str = "var_rta",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplicar la estrategia de split indicada.

    Punto de entrada unificado para seleccionar entre estrategias de forma
    intercambiable sin cambiar el código downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Modelo analítico completo.
    strategy : {"temporal", "random"}
        Estrategia de partición a aplicar.
    target_col : str
        Columna target para estratificación y logging.
    **kwargs
        Parámetros extra pasados a la estrategia seleccionada.
        Para "random": train_size, test_size, random_state.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_test, df_oot)

    Raises
    ------
    ValueError
        Si strategy no es "temporal" ni "random".
    """
    if strategy == "temporal":
        return temporal_split(df, target_col=target_col)
    elif strategy == "random":
        return random_split(df, target_col=target_col, **kwargs)
    else:
        raise ValueError(
            f"Estrategia desconocida: {strategy!r}. Opciones: 'temporal', 'random'"
        )


# ─── Utilidades internas ──────────────────────────────────────────────────────


def _log_partitions(
    strategy_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    target_col: str,
) -> None:
    total = len(df_train) + len(df_test) + len(df_oot)
    for label, part in (("train", df_train), ("test", df_test), ("oot", df_oot)):
        dist = (
            part[target_col].value_counts(normalize=True).round(3).to_dict()
            if target_col in part.columns else {}
        )
        if "f_analisis" in part.columns and len(part) > 0:
            periodos = sorted(part["f_analisis"].dt.strftime("%Y-%m-%d").unique().tolist())
            logger.info(
                "Split [%s] %-6s %5d filas (%.1f%%) | periodos: %s … %s | dist: %s",
                strategy_name, label, len(part), 100 * len(part) / total,
                periodos[0], periodos[-1], dist,
            )
        else:
            logger.info(
                "Split [%s] %-6s %5d filas (%.1f%%) | dist: %s",
                strategy_name, label, len(part), 100 * len(part) / total, dist,
            )
