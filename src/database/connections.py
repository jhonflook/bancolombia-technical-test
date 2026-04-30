"""Configuración de conexiones a la base de datos y MLflow — débitos recurrentes."""

from contextlib import contextmanager

import mlflow
from sqlmodel import Session, create_engine

from src.settings import settings

# Database engine
engine = create_engine(settings.debitdb_url)


@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# MLflow connection utilities


def setup_mlflow(experiment_name: str) -> mlflow.entities.Experiment:
    """Configure MLflow tracking and return experiment.

    Args:
        experiment_name: Name of the MLflow experiment to create or use.

    Returns:
        The MLflow Experiment object.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.set_experiment(experiment_name)


def get_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI from settings."""
    return settings.mlflow_tracking_uri
