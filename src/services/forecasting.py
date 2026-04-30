"""Servicio de scoring por lotes para obligaciones de débito recurrente.

Carga un clasificador entrenado (pkl) y genera probabilidades de débito
exclusivo para nuevas obligaciones. Asigna segmentos de cobranza (A/B/C/D)
según la definición operativa del negocio.

Uso standalone:
    uv run python -m src.services.forecasting \
        --model-path data/artifacts/model_xgboost.pkl \
        --data-path  data/artifacts/oot.parquet \
        --output     data/artifacts/scores_oot.parquet
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.database.connections import engine

logger = logging.getLogger(__name__)

JOIN_KEYS = ["num_doc", "obl17", "f_analisis"]
SCORE_TABLE = "debit_scores"
MORA_COL_6M = "moras_avg_mora_6m"


class DebitScoringService:
    """Scoring por lotes de obligaciones usando un clasificador entrenado.

    Parameters
    ----------
    threshold : float
        Umbral de clasificación binaria. Default 0.5.

    Attributes
    ----------
    model : BaseDebitClassifier or None
        Clasificador cargado desde pkl.
    feature_cols : list[str]
        Columnas de features requeridas por el modelo.
    threshold : float
        Umbral de decisión binaria.
    """

    def __init__(self, threshold: float = 0.5):
        self.model = None
        self.feature_cols: list[str] = []
        self.threshold = threshold

    def load_model(self, pkl_path: str | Path) -> "DebitScoringService":
        """Cargar modelo y feature_cols desde artefacto pkl.

        Parameters
        ----------
        pkl_path : str or Path
            Ruta al archivo pickle generado por train_and_evaluate().

        Returns
        -------
        DebitScoringService
            Self para encadenamiento de llamadas.
        """
        with open(pkl_path, "rb") as fh:
            artifact = pickle.load(fh)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        logger.info(
            "Modelo cargado | tipo=%s | features=%d | pkl=%s",
            type(self.model).__name__, len(self.feature_cols), pkl_path,
        )
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular probabilidades de débito exclusivo para un DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas de JOIN_KEYS y feature_cols.
            Nulos imputados con 0 internamente (S4/S5).

        Returns
        -------
        pd.DataFrame
            Columnas: num_doc, obl17, f_analisis (si existen), prob_debit,
            pred_label, segment. Ordenado por prob_debit descendente.
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado — llamar load_model() primero.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(
                "%d features faltantes → imputadas con 0: %s...",
                len(missing), missing[:5],
            )
            for col in missing:
                df[col] = 0.0

        X = df[self.feature_cols].fillna(0.0)
        prob = self.model.predict_proba(X)
        pred = (prob >= self.threshold).astype(int)

        mora_6m = (
            df[MORA_COL_6M].fillna(0.0)
            if MORA_COL_6M in df.columns
            else pd.Series(np.zeros(len(df)), index=df.index)
        )
        segments = self._assign_segments(pred, mora_6m.values)

        id_cols = [c for c in JOIN_KEYS if c in df.columns]
        result = df[id_cols].copy()
        result["prob_debit"] = np.round(prob, 6)
        result["pred_label"] = pred
        result["segment"] = segments

        return result.sort_values("prob_debit", ascending=False).reset_index(drop=True)

    @staticmethod
    def _assign_segments(pred_label: np.ndarray, mora_6m: np.ndarray) -> list[str]:
        """Asignar segmento de cobranza (A/B/C/D) por obligación.

        Segmentación operativa:
          A — Automatizar:        pred=1 y mora_6m ≤ 10 días
          B — Monitorear:         pred=1 y mora_6m > 10 días
          C — Cobranza suave:     pred=0 y mora_6m ≤ 15 días
          D — Cobranza intensiva: pred=0 y mora_6m > 15 días

        Parameters
        ----------
        pred_label : np.ndarray
            Predicciones binarias (0/1).
        mora_6m : np.ndarray
            Días de mora promedio en ventana de 6 meses.

        Returns
        -------
        list[str]
            Segmento por obligación: 'A', 'B', 'C' o 'D'.
        """
        return [
            ("A" if mora <= 10 else "B") if label == 1 else ("C" if mora <= 15 else "D")
            for label, mora in zip(pred_label, mora_6m)
        ]

    def score_from_parquet(
        self,
        parquet_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Cargar partición parquet, generar scores y opcionalmente persistir.

        Parameters
        ----------
        parquet_path : str or Path
            Ruta al parquet con obligaciones a scorear.
        output_path : str or Path or None
            Si se provee, guarda los scores como parquet.

        Returns
        -------
        pd.DataFrame
            Scores con prob_debit, pred_label, segment.
        """
        df = pd.read_parquet(parquet_path)
        logger.info("Cargando %d obligaciones desde %s", len(df), parquet_path)

        scores = self.score(df)

        logger.info(
            "Scoring completo | n=%d | pred=1: %d (%.1f%%) | segmentos=%s",
            len(scores),
            int(scores["pred_label"].sum()),
            100.0 * scores["pred_label"].mean(),
            scores["segment"].value_counts().to_dict(),
        )

        if output_path is not None:
            scores.to_parquet(output_path, index=False)
            logger.info("Scores guardados en %s", output_path)

        return scores

    def save_scores_to_db(
        self,
        scores: pd.DataFrame,
        if_exists: str = "append",
    ) -> None:
        """Persistir scores en tabla debit_scores de PostgreSQL.

        Parameters
        ----------
        scores : pd.DataFrame
            Scores generados por score().
        if_exists : str
            'append' (default) o 'replace' — pasado a pandas.to_sql().
        """
        scores.to_sql(
            SCORE_TABLE,
            con=engine,
            if_exists=if_exists,
            index=False,
            method="multi",
        )
        logger.info(
            "Scores persistidos en '%s' | n=%d filas",
            SCORE_TABLE, len(scores),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scoring de débitos recurrentes")
    parser.add_argument("--model-path", required=True, help="Ruta al pkl del modelo")
    parser.add_argument("--data-path",  required=True, help="Parquet con obligaciones a scorear")
    parser.add_argument("--output",     default=None,  help="Ruta de salida (parquet)")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Umbral de clasificación")
    args = parser.parse_args()

    service = DebitScoringService(threshold=args.threshold)
    service.load_model(args.model_path)
    service.score_from_parquet(args.data_path, output_path=args.output)
