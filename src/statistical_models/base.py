"""Clasificador base para débitos recurrentes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ClassificationMetrics:
    """Métricas de clasificación binaria — alineadas con S8 (AUC + KS + Precision/Recall).

    Attributes
    ----------
    partition : str
        Nombre de la partición evaluada: 'train', 'test' o 'oot'.
    n_samples : int
        Total de registros en la partición.
    n_positive : int
        Registros de clase 1 (débito exclusivo + recurrencia >= 40%).
    prevalence : float
        Proporción de clase 1 sobre el total.
    auc_roc : float
        Área bajo la curva ROC (métrica primaria S8).
    ks_stat : float
        Estadístico KS: máxima separación TPR − FPR.
    ks_threshold : float
        Umbral de probabilidad donde se maximiza el KS.
    precision_05 : float
        Precisión con umbral fijo 0.5.
    recall_05 : float
        Recall con umbral fijo 0.5.
    f1_05 : float
        F1 con umbral fijo 0.5.
    precision_ks : float
        Precisión con umbral óptimo KS.
    recall_ks : float
        Recall con umbral óptimo KS.
    f1_ks : float
        F1 con umbral óptimo KS.
    auc_pr : float
        Área bajo la curva Precision-Recall (relevante con desbalance 3.7:1).
    report : str
        Reporte completo de clasificación (sklearn).
    """

    partition: str
    n_samples: int
    n_positive: int
    prevalence: float
    auc_roc: float
    ks_stat: float
    ks_threshold: float
    precision_05: float
    recall_05: float
    f1_05: float
    precision_ks: float
    recall_ks: float
    f1_ks: float
    auc_pr: float
    report: str = field(default="", repr=False)


class BaseDebitClassifier(ABC):
    """Clase base abstracta para clasificadores de débito recurrente.

    Todos los clasificadores deben heredar de esta clase e implementar
    los métodos abstractos. Garantiza interfaz uniforme para MODEL_REGISTRY
    y el pipeline de comparación multi-modelo.

    Attributes
    ----------
    name : str
        Identificador único del tipo de modelo (clave en MODEL_REGISTRY).
    requires_scaling : bool
        Si True, los features se estandarizan con StandardScaler antes del ajuste.
    """

    name: str
    requires_scaling: bool = False

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.selected_features: list[str] = []
        self.is_fitted: bool = False

    @abstractmethod
    def _create_model(self, **params: Any) -> Any:
        """Crear modelo subyacente sklearn-compatible con los parámetros dados.

        Parameters
        ----------
        **params : Any
            Hiperparámetros del modelo incluyendo scale_pos_weight.

        Returns
        -------
        Any
            Instancia del estimador sklearn-compatible.
        """

    @abstractmethod
    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Definir espacio de búsqueda de hiperparámetros para Optuna.

        Parameters
        ----------
        trial : optuna.Trial
            Trial de Optuna para sugerir valores.

        Returns
        -------
        dict[str, Any]
            Diccionario de hiperparámetros a valores sugeridos.
        """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: list[str],
        scale_pos_weight: float = 1.0,
        **params: Any,
    ) -> Self:
        """Ajustar el clasificador sobre datos de entrenamiento.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame de features.
        y : pd.Series
            Serie target binaria (0 = otros canales, 1 = débito exclusivo).
        selected_features : list[str]
            Lista de columnas a utilizar en el ajuste.
        scale_pos_weight : float
            Cociente n_neg/n_pos para corrección de desbalance (S8).
        **params : Any
            Hiperparámetros adicionales del modelo.

        Returns
        -------
        Self
            Self para encadenamiento de métodos.
        """
        self.selected_features = selected_features
        X_sel = X[selected_features]

        if self.requires_scaling:
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X_sel)
        else:
            X_proc = X_sel.values

        self.model = self._create_model(scale_pos_weight=scale_pos_weight, **params)
        self.model.fit(X_proc, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predecir probabilidad de clase 1 (débito exclusivo recurrente).

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame que contiene las columnas en selected_features.

        Returns
        -------
        np.ndarray
            Array 1-D de probabilidades para la clase positiva.

        Raises
        ------
        ValueError
            Si el modelo no ha sido ajustado previamente.
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ajustarse antes de predecir.")

        X_sel = X[self.selected_features]

        if self.requires_scaling and self.scaler is not None:
            X_proc = self.scaler.transform(X_sel)
        else:
            X_proc = X_sel.values

        return self.model.predict_proba(X_proc)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predecir clases binarias con umbral configurable.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame que contiene las columnas en selected_features.
        threshold : float
            Umbral de probabilidad para asignar clase 1 (default 0.5).

        Returns
        -------
        np.ndarray
            Array 1-D de predicciones binarias (0 / 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> dict[str, Any]:
        """Devolver parámetros del modelo ajustado.

        Returns
        -------
        dict[str, Any]
            Parámetros del estimador interno, o dict vacío si no ajustado.
        """
        if self.model is None:
            return {}
        return self.model.get_params()

    def get_training_history(self) -> dict | None:
        """Devolver historial de métricas por iteración del entrenamiento, si está disponible.

        Returns
        -------
        dict | None
            Diccionario con arrays de métricas por paso, o None si no disponible.
        """
        return None

    def get_feature_importances(self) -> dict[str, float] | None:
        """Devolver importancia de features si el estimador la expone.

        Returns
        -------
        dict[str, float] | None
            Mapeo feature → importancia, o None si no disponible.
        """
        if not self.is_fitted or self.model is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.selected_features, self.model.feature_importances_))

        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return dict(zip(self.selected_features, np.abs(coef)))

        return None
