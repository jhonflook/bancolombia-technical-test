"""Random Forest regression model implementation."""

from typing import Any

import optuna
from sklearn.ensemble import RandomForestRegressor

from src.statistical_models.base import BaseStatisticalModel


class RandomForestModel(BaseStatisticalModel):
    """Random Forest regression model.

    Random Forest is an ensemble method that fits multiple decision trees
    on various sub-samples and uses averaging to improve predictive
    accuracy and control overfitting.
    """

    name = "random_forest"
    requires_scaling = False

    def _create_model(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        **kwargs: Any,
    ) -> RandomForestRegressor:
        """Create Random Forest regression model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            **kwargs: Additional arguments (ignored).

        Returns:
            Configured RandomForestRegressor model.
        """
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for Random Forest.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with tree-related hyperparameters.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
