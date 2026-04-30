"""Hyperparameter optimization service with Optuna and MLflow integration."""

import logging
from typing import Any

import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.dataset import FeatureConfig
from src.models import TargetType
from src.statistical_models import MODEL_REGISTRY, evaluate_model_cv

logger = logging.getLogger(__name__)


class MLflowTrialCallback:
    """Optuna callback to log trial metrics to MLflow with step parameter."""

    def __init__(self, model_name: str):
        """Initialize callback with model name prefix for metric keys.

        Args:
            model_name: Model name used as prefix for logged metrics.
        """
        self.model_name = model_name

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Log trial metrics to MLflow after each trial completes.

        Args:
            study: The Optuna study object.
            trial: The completed trial.
        """
        # Get metrics from trial user attributes
        rmse = trial.user_attrs.get("rmse", trial.value)
        r2 = trial.user_attrs.get("r2", 0.0)
        n_features = trial.user_attrs.get("n_features", 0)

        # Log metrics with step=trial.number for visualization
        mlflow.log_metric(f"{self.model_name}_trial_rmse", rmse, step=trial.number)
        mlflow.log_metric(f"{self.model_name}_trial_r2", r2, step=trial.number)
        mlflow.log_metric(
            f"{self.model_name}_trial_n_features", n_features, step=trial.number
        )

        # Track best values so far
        best_rmse = study.best_value if study.best_trial else rmse
        mlflow.log_metric(
            f"{self.model_name}_best_rmse_so_far", best_rmse, step=trial.number
        )


def create_objective_with_feature_selection(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    base_feature: str,
    candidate_features: list[str],
) -> Any:
    """Create Optuna objective that jointly optimizes hyperparameters and feature selection.

    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY).
        X: Feature DataFrame.
        y: Target Series.
        cv: TimeSeriesSplit cross-validator.
        base_feature: Feature always included (target_t_1).
        candidate_features: Features to consider adding.

    Returns:
        Objective function for Optuna optimization.
    """
    model_class = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        # Feature selection - binary decision for each candidate feature
        selected_features = [base_feature]

        for feature in candidate_features:
            if feature in X.columns:
                include = trial.suggest_categorical(f"include_{feature}", [True, False])
                if include:
                    selected_features.append(feature)

        X_selected = X[selected_features]

        # Get model instance and hyperparameters
        model_instance = model_class()
        params = model_instance.get_hyperparameter_space(trial)

        # Create the underlying sklearn model
        model = model_instance._create_model(**params)

        # Evaluate with cross-validation
        results = evaluate_model_cv(
            model, X_selected, y, cv, scale_features=model_class.requires_scaling
        )

        # Store additional info in trial
        trial.set_user_attr("n_features", len(selected_features))
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("rmse", results["mean_rmse"])
        trial.set_user_attr("r2", results["mean_r2"])

        return results["mean_rmse"]

    return objective


class HyperparameterOptimizer:
    """Handles hyperparameter optimization with joint feature selection."""

    def __init__(
        self,
        model_name: str,
        feature_config: FeatureConfig,
        n_trials: int = 30,
    ):
        """Initialize the optimizer.

        Args:
            model_name: Name of the model to optimize.
            feature_config: Configuration specifying available features.
            n_trials: Number of Optuna trials to run.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_name = model_name
        self.model_class = MODEL_REGISTRY[model_name]
        self.feature_config = feature_config
        self.n_trials = n_trials

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit,
        coin_id: str,
        target_type: TargetType,
    ) -> tuple[optuna.Study, dict[str, Any], list[str]]:
        """Run optimization and return study, best params, selected features.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            cv: TimeSeriesSplit cross-validator.
            coin_id: Identifier for the coin being modeled.
            target_type: Target variable type.

        Returns:
            Tuple of (Optuna study, best hyperparameters, selected features).
        """
        with mlflow.start_run(
            run_name=f"{self.model_name}_{coin_id}_optimization", nested=True
        ):
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("coin_id", coin_id)
            mlflow.log_param("target_type", target_type)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("n_features_available", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("cv_splits", cv.n_splits)

            # Get feature selection config
            base_feature = self.feature_config.base_feature
            candidate_features = self.feature_config.get_candidate_features()
            candidate_features = [f for f in candidate_features if f in X.columns]

            # Create joint optimization objective
            objective = create_objective_with_feature_selection(
                model_name=self.model_name,
                X=X,
                y=y,
                cv=cv,
                base_feature=base_feature,
                candidate_features=candidate_features,
            )

            study = optuna.create_study(
                direction="minimize",
                study_name=f"{self.model_name}_{coin_id}",
            )

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Create callback for trial-by-trial MLflow logging
            mlflow_callback = MLflowTrialCallback(self.model_name)

            study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
                callbacks=[mlflow_callback],
            )

            # Extract selected features from best trial
            selected_features = study.best_trial.user_attrs.get(
                "selected_features", [base_feature]
            )
            best_r2 = study.best_trial.user_attrs.get("r2", 0.0)

            # Extract hyperparameters only (exclude include_* feature params)
            best_params = {
                k: v
                for k, v in study.best_params.items()
                if not k.startswith("include_")
            }

            # Log final results
            mlflow.log_metric("best_rmse", study.best_value)
            mlflow.log_metric("best_r2", best_r2)
            mlflow.log_param("n_selected_features", len(selected_features))
            mlflow.log_param("selected_features", ",".join(selected_features))
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

            unit = "%" if target_type == "pct_change" else "$"
            logger.info(
                f"{self.model_name} optimization complete: "
                f"Best RMSE = {study.best_value:.4f}{unit}, "
                f"Selected {len(selected_features)}/{len(candidate_features)+1} features"
            )

            return study, best_params, selected_features
