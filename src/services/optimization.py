"""Selección y comparación de clasificadores desde MLflow.

Consulta el experimento debit_recurrence_classifier para comparar
runs anidados y determinar el mejor modelo según métricas en Test/OOT.

Uso standalone:
    uv run python -m src.services.optimization \
        --mlflow-uri http://localhost:5000 \
        --metric test_auc_roc
"""

import argparse
import logging

import mlflow
import mlflow.tracking
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "debit-models"
DEFAULT_METRIC = "test_auc_roc"


class ModelSelectionService:
    """Comparación y selección del mejor clasificador desde MLflow.

    Parameters
    ----------
    tracking_uri : str
        URI del servidor MLflow.
    experiment_name : str
        Nombre del experimento MLflow a consultar.

    Attributes
    ----------
    tracking_uri : str
    experiment_name : str
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = EXPERIMENT_NAME,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)

    def get_runs_summary(self, metric: str = DEFAULT_METRIC) -> pd.DataFrame:
        """Obtener resumen de runs ordenado por métrica descendente.

        Parameters
        ----------
        metric : str
            Métrica de comparación. Default: test_auc_roc.

        Returns
        -------
        pd.DataFrame
            Columnas: run_id, run_name, model_type, métricas Train/Test/OOT.
            Vacío si el experimento no existe o no tiene runs.
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.warning("Experimento '%s' no encontrado en MLflow.", self.experiment_name)
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} DESC"],
        )
        if not runs:
            logger.warning("Sin runs en '%s'.", self.experiment_name)
            return pd.DataFrame()

        metrics_keys = [
            "train_auc_roc", "test_auc_roc", "oot_auc_roc",
            "train_ks",      "test_ks",      "oot_ks",
            "test_f1_05",    "oot_f1_05",
            "test_auc_pr",   "oot_auc_pr",
        ]
        records = []
        for run in runs:
            row: dict = {
                "run_id":     run.info.run_id,
                "run_name":   run.info.run_name,
                "status":     run.info.status,
                "model_type": run.data.params.get("model_type", "—"),
            }
            for k in metrics_keys:
                row[k] = run.data.metrics.get(k)
            records.append(row)

        df = pd.DataFrame(records)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        return df

    def get_best_run(self, metric: str = DEFAULT_METRIC) -> dict:
        """Obtener el run con mejor valor de la métrica especificada.

        Parameters
        ----------
        metric : str
            Métrica de selección.

        Returns
        -------
        dict
            run_id, run_name, model_type y valor de la métrica. Vacío si no hay runs.
        """
        df = self.get_runs_summary(metric=metric)
        if df.empty:
            return {}

        best = df.iloc[0]
        result = {
            "run_id":     best["run_id"],
            "run_name":   best["run_name"],
            "model_type": best.get("model_type", "—"),
            metric:       best.get(metric),
        }
        logger.info(
            "Mejor modelo: %s | run_id=%s | %s=%.4f",
            result["model_type"],
            result["run_id"],
            metric,
            result.get(metric) or 0.0,
        )
        return result

    def print_comparison_table(self, metric: str = DEFAULT_METRIC) -> None:
        """Imprimir tabla comparativa de modelos en el log.

        Parameters
        ----------
        metric : str
            Métrica principal de comparación.
        """
        df = self.get_runs_summary(metric=metric)
        if df.empty:
            logger.warning("Sin datos para comparar.")
            return

        cols = [
            "run_name", "model_type",
            "train_auc_roc", "test_auc_roc", "oot_auc_roc",
            "test_ks", "oot_ks",
        ]
        display = [c for c in cols if c in df.columns]

        logger.info("=" * 90)
        logger.info("COMPARACIÓN DE MODELOS — %s", self.experiment_name)
        logger.info("Ordenado por: %s", metric)
        logger.info("=" * 90)
        for _, row in df[display].iterrows():
            logger.info(
                "%-28s %-22s auc: train=%.4f test=%.4f oot=%.4f | ks: test=%.4f oot=%.4f",
                str(row.get("run_name", ""))[:28],
                str(row.get("model_type", ""))[:22],
                row.get("train_auc_roc") or 0.0,
                row.get("test_auc_roc") or 0.0,
                row.get("oot_auc_roc") or 0.0,
                row.get("test_ks") or 0.0,
                row.get("oot_ks") or 0.0,
            )
        logger.info("=" * 90)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Comparación y selección de modelos — débitos recurrentes"
    )
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Métrica de selección")
    args = parser.parse_args()

    svc = ModelSelectionService(tracking_uri=args.mlflow_uri)
    svc.print_comparison_table(metric=args.metric)
    best = svc.get_best_run(metric=args.metric)
    if best:
        print(f"\nMejor modelo: {best['model_type']} | run_id={best['run_id']}")
