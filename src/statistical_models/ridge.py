"""Ridge regression model implementation."""

from typing import Any

import optuna
from sklearn.linear_model import Ridge

from src.statistical_models.base import BaseStatisticalModel


class RidgeModel(BaseStatisticalModel):
    """Ridge regression model with L2 regularization.

    Ridge regression is a linear model that uses L2 regularization
    to prevent overfitting. It's effective when features are correlated.
    """

    name = "ridge"
    requires_scaling = True

    def _create_model(self, alpha: float = 1.0, **kwargs: Any) -> Ridge:
        """Create Ridge regression model.

        Args:
            alpha: Regularization strength. Larger values = more regularization.
            **kwargs: Additional arguments (ignored).

        Returns:
            Configured Ridge model.
        """
        return Ridge(alpha=alpha)

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for Ridge.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with 'alpha' hyperparameter.
        """
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100, log=True),
        }
