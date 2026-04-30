#!/usr/bin/env python3
"""
Train cryptocurrency price prediction models using percentage change approach.

This script trains ML models for bitcoin, ethereum, and cardano using:
- MLflow for experiment tracking
- Optuna for hyperparameter optimization
- Time-series cross-validation
- Saves forecasts (always as price_usd) to the database

Target options:
- 'price_usd': Direct price prediction
- 'pct_change': Predict daily percentage change, then convert to price

Available models:
- ridge, elasticnet, random_forest, gradient_boosting (sklearn-based)
- sarimax, prophet (time series models)

Usage:
    uv run python deploy/train_debit_classifier.py
    uv run python deploy/train_debit_classifier.py --target pct_change
    uv run python deploy/train_debit_classifier.py --coins bitcoin ethereum --target pct_change
    uv run python deploy/train_debit_classifier.py --n-trials 50 --forecast-days 15 --target price_usd
    uv run python deploy/train_debit_classifier.py --models ridge elasticnet prophet
"""

import argparse
import logging
from datetime import date, timedelta

import mlflow
import pandas as pd
from sqlmodel import Session, create_engine

from src.database import get_mlflow_tracking_uri, setup_mlflow
from src.database.crud import get_coin_data
from src.dataset import engineer_all_features
from src.models import TargetType
from src.services import ForecastingService, TrainingService
from src.settings import settings
from src.statistical_models import MODEL_REGISTRY

# Default models to train (sklearn-based, more stable for optimization)
DEFAULT_MODELS = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train cryptocurrency price prediction models"
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        default=["bitcoin", "ethereum", "cardano"],
        help="Coins to train models for",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["price_usd", "pct_change"],
        default="pct_change",
        help="Target variable: 'price_usd' or 'pct_change' (default: pct_change)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=15,
        help="Number of days to forecast",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Models to train. Available: {', '.join(MODEL_REGISTRY.keys())}. "
        f"Default: {', '.join(DEFAULT_MODELS)}",
    )
    return parser.parse_args()


def load_coin_data(
    coins: list[str], start_date: date, end_date: date
) -> dict[str, pd.DataFrame]:
    """Load data for all coins from database."""
    engine = create_engine(settings.cryptodb_url)
    session = Session(engine)

    coin_data = {}
    for coin in coins:
        data = get_coin_data(session, coin, start_date, end_date)
        df = pd.DataFrame(data.model_dump().get("coin_dataset"))
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        coin_data[coin] = df
        logger.info(
            f"Loaded {coin}: {len(df)} records "
            f"({df['date'].min().date()} to {df['date'].max().date()})"
        )

    session.close()
    return coin_data


def log_parent_run_metrics(all_metrics: dict, best_model_name: str, best_result) -> None:
    """Log comprehensive metrics at parent run level for MLflow UI."""
    # Optimization metrics (CV RMSE for each model)
    for model_name, opt_metrics in all_metrics["optimization"].items():
        mlflow.log_metric(f"{model_name}_cv_rmse", opt_metrics["best_cv_rmse"])
        mlflow.log_metric(f"{model_name}_cv_r2", opt_metrics["best_cv_r2"])

    # Validation metrics (train/test for each model)
    for model_name, val_metrics in all_metrics["validation"].items():
        mlflow.log_metric(f"{model_name}_train_rmse", val_metrics["train_rmse"])
        mlflow.log_metric(f"{model_name}_test_rmse", val_metrics["test_rmse"])
        mlflow.log_metric(f"{model_name}_train_mae", val_metrics["train_mae"])
        mlflow.log_metric(f"{model_name}_test_mae", val_metrics["test_mae"])
        mlflow.log_metric(f"{model_name}_train_r2", val_metrics["train_r2"])
        mlflow.log_metric(f"{model_name}_test_r2", val_metrics["test_r2"])

    # Feature selection metrics
    for model_name, feat_metrics in all_metrics["feature_selection"].items():
        mlflow.log_metric(f"{model_name}_n_features", feat_metrics["n_features"])

    # Best model summary
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric(
        "best_model_cv_rmse",
        all_metrics["optimization"][best_model_name]["best_cv_rmse"],
    )
    mlflow.log_metric("best_model_test_rmse", best_result.metrics.test_rmse)
    mlflow.log_metric("best_model_test_mae", best_result.metrics.test_mae)
    mlflow.log_metric("best_model_test_r2", best_result.metrics.test_r2)
    mlflow.log_metric("best_model_train_rmse", best_result.metrics.train_rmse)
    mlflow.log_metric("best_model_train_r2", best_result.metrics.train_r2)
    mlflow.log_metric(
        "best_model_n_features",
        all_metrics["feature_selection"][best_model_name]["n_features"],
    )


def main():
    """Main entry point for training forecast models."""
    args = parse_args()
    target_type: TargetType = args.target
    model_names = args.models if args.models else DEFAULT_MODELS

    # Setup MLflow
    experiment = setup_mlflow(f"crypto_price_prediction_{target_type}")

    logger.info("=" * 70)
    logger.info("CRYPTOCURRENCY PRICE PREDICTION TRAINING")
    logger.info("=" * 70)
    logger.info(f"MLflow Tracking URI: {get_mlflow_tracking_uri()}")
    logger.info(f"Experiment: {experiment.name}")
    logger.info(f"Target type: {target_type}")
    logger.info(f"Coins: {args.coins}")
    logger.info(f"Models: {model_names}")
    logger.info(f"Optuna trials per model: {args.n_trials}")
    logger.info(f"Forecast horizon: {args.forecast_days} days")

    # Date range
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else end_date - timedelta(days=365)
    )
    logger.info(f"Training data range: {start_date} to {end_date}")

    # Load and prepare data
    coin_data = load_coin_data(args.coins, start_date, end_date)

    logger.info(f"Engineering features (target_type={target_type})...")
    for coin in args.coins:
        coin_data[coin] = engineer_all_features(coin_data[coin], target_type)

    # Initialize services
    training_service = TrainingService(
        n_trials=args.n_trials,
        target_type=target_type,
        model_names=model_names,
    )
    forecasting_service = ForecastingService(target_type=target_type)

    # Train models for each coin
    results = {}

    for coin in args.coins:
        logger.info("=" * 70)
        logger.info(f"TRAINING {coin.upper()} (target: {target_type})")
        logger.info("=" * 70)

        with mlflow.start_run(run_name=f"{coin}_{target_type}_training"):
            parent_run_id = mlflow.active_run().info.run_id
            mlflow.log_param("coin_id", coin)
            mlflow.log_param("target_type", target_type)
            mlflow.log_param("models", ",".join(model_names))
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("forecast_days", args.forecast_days)
            mlflow.log_param("start_date", str(start_date))
            mlflow.log_param("end_date", str(end_date))

            # Train all models and get best
            best_result, predictions, best_model_name, all_metrics = (
                training_service.train_coin_models(
                    coin_id=coin,
                    df=coin_data[coin],
                    forecast_days=args.forecast_days,
                )
            )

            results[coin] = {
                "best_model": best_model_name,
                "metrics": best_result.metrics,
                "predictions": predictions,
                "run_id": best_result.run_id,
                "parent_run_id": parent_run_id,
                "selected_features": best_result.selected_features,
            }

            # Log metrics at parent level
            log_parent_run_metrics(all_metrics, best_model_name, best_result)

            # Log and save predictions
            with mlflow.start_run(run_name=f"{coin}_predictions", nested=True) as pred_run:
                mlflow.log_param("coin_id", coin)
                mlflow.log_param("target_type", target_type)
                mlflow.log_param("model_used", best_model_name)
                mlflow.log_param("forecast_days", args.forecast_days)
                mlflow.log_metric("predicted_price_min", predictions["predicted_price"].min())
                mlflow.log_metric("predicted_price_max", predictions["predicted_price"].max())
                mlflow.log_metric("predicted_price_mean", predictions["predicted_price"].mean())

                # Save predictions to database
                forecasting_service.save_predictions_to_db(
                    predictions=predictions,
                    coin_id=coin,
                    run_id=pred_run.info.run_id,
                )

                # Save forecast metadata to database
                forecasting_service.save_metadata_to_db(
                    forecast_id=pred_run.info.run_id,
                    coin_id=coin,
                    model_type=best_model_name,
                    mape=best_result.metrics.test_mape,
                    rmse=best_result.metrics.test_rmse,
                    r2=best_result.metrics.test_r2,
                    mae=best_result.metrics.test_mae,
                    n_train_samples=best_result.metrics.n_train_samples,
                    n_test_samples=best_result.metrics.n_test_samples,
                    n_trials=args.n_trials,
                    n_features=len(best_result.selected_features),
                    selected_features=best_result.selected_features,
                    forecast_days=args.forecast_days,
                    train_start_date=start_date,
                    train_end_date=end_date,
                    mlflow_run_id=parent_run_id,
                    tag="unofficial",
                )

    # Print summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Target type used: {target_type}")

    unit = "%" if target_type == "pct_change" else "$"

    for coin, result in results.items():
        logger.info(f"\n{coin.upper()}:")
        logger.info(f"  Best Model: {result['best_model']}")
        logger.info(f"  Test RMSE: {result['metrics'].test_rmse:.4f}{unit}")
        logger.info(f"  Test MAE: {result['metrics'].test_mae:.4f}{unit}")
        logger.info(f"  Test MAPE: {result['metrics'].test_mape:.2f}%")
        logger.info(f"  Test R2: {result['metrics'].test_r2:.4f}")
        logger.info(
            f"  Samples: {result['metrics'].n_train_samples} train, "
            f"{result['metrics'].n_test_samples} test"
        )
        logger.info(
            f"  Predictions: {result['predictions']['date'].min().date()} "
            f"to {result['predictions']['date'].max().date()}"
        )
        logger.info(
            f"  Price Range (USD): ${result['predictions']['predicted_price'].min():,.2f} "
            f"- ${result['predictions']['predicted_price'].max():,.2f}"
        )
        logger.info(f"  MLflow Run ID: {result['parent_run_id']}")

    logger.info(f"\nView results at: {get_mlflow_tracking_uri()}")


if __name__ == "__main__":
    main()
