"""Base class for statistical models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    train_mape: float = 0.0
    test_mape: float = 0.0
    n_train_samples: int = 0
    n_test_samples: int = 0


class BaseStatisticalModel(ABC):
    """Abstract base class for all statistical models.

    All model implementations must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across
    different model types.

    Attributes:
        name: Unique identifier for the model type.
        requires_scaling: Whether features should be standardized before fitting.
    """

    name: str
    requires_scaling: bool

    def __init__(self):
        """Initialize the model."""
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.selected_features: list[str] = []
        self.is_fitted: bool = False

    @abstractmethod
    def _create_model(self, **params: Any) -> Any:
        """Create the underlying sklearn model with given parameters.

        Args:
            **params: Model-specific hyperparameters.

        Returns:
            The sklearn model instance.
        """
        pass

    @abstractmethod
    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define Optuna hyperparameter search space.

        Args:
            trial: Optuna trial object for suggesting hyperparameters.

        Returns:
            Dictionary of hyperparameter names to suggested values.
        """
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: list[str],
        **params: Any,
    ) -> Self:
        """Fit the model on training data.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            selected_features: List of feature names to use.
            **params: Model-specific hyperparameters.

        Returns:
            Self for method chaining.
        """
        self.selected_features = selected_features
        X_selected = X[selected_features]

        if self.requires_scaling:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_selected)
        else:
            X_processed = X_selected.values

        self.model = self._create_model(**params)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature DataFrame (must contain selected_features columns).

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_selected = X[self.selected_features]

        if self.requires_scaling and self.scaler is not None:
            X_processed = self.scaler.transform(X_selected)
        else:
            X_processed = X_selected.values

        return self.model.predict(X_processed)

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of parameter names to values.
        """
        if self.model is None:
            return {}
        return self.model.get_params()

    def get_feature_importances(self) -> dict[str, float] | None:
        """Get feature importances if available.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available.
        """
        if not self.is_fitted or self.model is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.selected_features, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            return dict(zip(self.selected_features, np.abs(self.model.coef_)))
        return None
