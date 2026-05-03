## Project Structure

```
.
├── alembic/
│   ├── versions/           # Database migration files
│   └── env.py              # Alembic environment config
├── data/
│   ├── dags/               # Airflow DAG files
│   ├── queries/            # Downloaded coin data (JSON files)
│   └── coin_data.sql       # Historical coin data SQL dump
├── deploy/
│   ├── Dockerfile          # Docker image for deploy scripts
│   ├── entrypoint.sh       # Runs migrations on container start
│   ├── coin_fetcher.py     # CLI app for fetching coin data
│   └── train_forecast_models_pct.py  # ML training entry point
├── docker/
│   └── postgres/
│       └── init-databases.sh  # Creates cryptodb database
├── src/
│   ├── database/
│   │   ├── connections.py  # Database + MLflow connection management
│   │   └── crud/           # CRUD operations
│   ├── dataset/            # Feature engineering & data preparation
│   │   ├── feature_config.py
│   │   ├── feature_engineering.py
│   │   └── data_preparation.py
│   ├── models/             # SQLModel table definitions
│   ├── services/           # High-level service orchestration
│   │   ├── training.py     # TrainingService
│   │   ├── optimization.py # HyperparameterOptimizer
│   │   └── forecasting.py  # ForecastingService
│   ├── statistical_models/ # ML model implementations
│   │   ├── base.py         # BaseStatisticalModel ABC
│   │   ├── ridge.py
│   │   ├── elasticnet.py
│   │   ├── random_forest.py
│   │   ├── gradient_boosting.py
│   │   ├── sarimax.py      # SARIMAX time series model
│   │   └── prophet_model.py # Facebook Prophet model
│   └── settings.py         # Pydantic settings configuration
├── scripts/
│   └── export_pdf.py       # PDF export script with cover page
├── templates/
│   └── custom_report/      # Custom nbconvert template
│       ├── assets/         # Cover page image
│       ├── conf.json
│       └── index.tex.j2
├── docker-compose.yml      # Airflow + PostgreSQL Docker setup
├── pyproject.toml          # Python project configuration (UV)
└── .env                    # Environment variables
```

## Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables in `.env`:
```
COINGECKO_API_KEY='your-api-key-here'
```

## CLI Usage

### Basic Commands

Fetch coin data for a specific date:
```bash
uv run python deploy/coin_fetcher.py 2025-01-01 bitcoin
```

Fetch coin data for a date range:
```bash
uv run python deploy/coin_fetcher.py bitcoin --start 2025-01-01 --end 2025-01-05
```

### Options

| Option | Description |
|--------|-------------|
| `--store-db` | Store data in PostgreSQL database |
| `--no-file` | Skip saving to JSON file |
| `--start` | Bulk start date (YYYY-MM-DD) |
| `--end` | Bulk end date (YYYY-MM-DD) |
| `--force` | Force API fetch, ignore cached files |

### Examples

Fetch and save to JSON only (default):
```bash
uv run python deploy/coin_fetcher.py 2025-01-01 bitcoin
```

Fetch and store in database:
```bash
uv run python deploy/coin_fetcher.py 2025-01-01 bitcoin --store-db
```

Store in database without saving JSON file:
```bash
uv run python deploy/coin_fetcher.py 2025-01-01 bitcoin --store-db --no-file
```

Bulk fetch with database storage:
```bash
uv run python deploy/coin_fetcher.py bitcoin --start 2025-01-01 --end 2025-01-31 --store-db
```

Data is saved to `data/queries/{coin}_{date}.json`.

## Database

The project uses PostgreSQL with three databases:
- **airflow** - Airflow metadata
- **cryptodb** - Coin data storage
- **metabaseappdb** - Metabase application data

### Tables

| Table | Description |
|-------|-------------|
| `coin_data` | Daily coin prices with full JSON response |
| `coin_monthly_aggregation` | Monthly max/min price summaries |
| `forecast_data` | ML model predictions with confidence bounds |

### Running Migrations

Migrations run automatically when the coin-fetcher container starts. To run manually:

```bash
uv run alembic upgrade head
```

## Airflow Setup

The project includes an Airflow setup via Docker Compose for automated daily data fetching.

### Starting Airflow

1. Set required environment variables:
```bash
export COINGECKO_API_KEY=$(grep COINGECKO_API_KEY .env | cut -d "'" -f2)
export PROJECT_ROOT=$(pwd)
```

2. Build and start the Airflow services:
```bash
docker compose build coin-fetcher
docker compose up -d
```

3. Access the Airflow web UI at http://localhost:8080
   - Username: `admin`
   - Password: `admin`

### Stopping Airflow

```bash
docker compose down
```

To remove all data (volumes):
```bash
docker compose down -v
```

### DAG: coin_fetcher_daily

The `coin_fetcher_daily` DAG runs automatically at **3:00 AM UTC daily** and fetches historical data for:
- Bitcoin
- Ethereum
- Cardano

| Setting | Value |
|---------|-------|
| Schedule | `0 3 * * *` (3am UTC daily) |
| Coins | bitcoin, ethereum, cardano |
| Output | JSON files + PostgreSQL database |

The DAG fetches the previous day's data from CoinGecko API, stores it as JSON files, and persists it to the PostgreSQL database with monthly aggregations.

### Adding Custom DAGs

Place new DAG files in the `data/dags/` folder. They will be automatically picked up by Airflow.

## ML Forecast Models

The project includes two scripts for training cryptocurrency price prediction models using MLflow for experiment tracking and Optuna for hyperparameter optimization.

### Approach 1: Direct Price Prediction (`train_forecast_models.py`)

This approach predicts the next day's price directly using historical price data as features.

```bash
# Train models for all coins (bitcoin, ethereum, cardano)
uv run python deploy/train_forecast_models.py

# Train for specific coins
uv run python deploy/train_forecast_models.py --coins bitcoin ethereum cardano --n-trials 50 --forecast-days 15

# Customize training parameters
uv run python deploy/train_forecast_models.py --n-trials 50 --forecast-days 15

# Specify date range for training data
uv run python deploy/train_forecast_models.py --start-date 2024-01-01 --end-date 2024-12-31
```

**Features used:**
- Price lags: `price_t_1` to `price_t_7` (previous 7 days' prices)
- Return lags: `return_t_1` to `return_t_3` (previous days' percentage changes)
- Rolling statistics: mean, std, skew, kurtosis over 7/14/30 day windows
- Technical indicators: momentum, RSI, trend encoding
- Calendar features: day of week, month, holidays, cyclical encodings

### Approach 2: Percentage Change Prediction (`train_forecast_models_pct.py`)

This approach predicts the daily percentage change (returns) instead of absolute prices. The predicted percentage change is then converted back to price for storage.

```bash
# Train using pct_change as target (default)
uv run python deploy/train_forecast_models_pct.py

# Explicitly specify target type
uv run python deploy/train_forecast_models_pct.py --target pct_change

# Use price_usd as target (same as Approach 1)
uv run python deploy/train_forecast_models_pct.py --target price_usd

# Full example with all options
uv run python deploy/train_forecast_models_pct.py --coins bitcoin ethereum cardano --target pct_change --n-trials 50 --forecast-days 15

# Train specific models only
uv run python deploy/train_forecast_models_pct.py --models ridge prophet

# Train time series models only
uv run python deploy/train_forecast_models_pct.py --models sarimax prophet

# Train all available models
uv run python deploy/train_forecast_models_pct.py --models ridge elasticnet random_forest gradient_boosting sarimax prophet
```

**Key differences from Approach 1:**
- Target variable: daily percentage change instead of absolute price
- Lag features: uses lags of pct_change (`target_t_1` to `target_t_7`)
- Conversion: predictions are converted back to `price_usd` before saving to database
- Stationarity: percentage changes are typically more stationary than raw prices

### Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--coins` | Coins to train models for | `bitcoin ethereum cardano` |
| `--target` | Target variable (pct_change script only) | `pct_change` |
| `--models` | Models to train (see below) | `ridge elasticnet random_forest gradient_boosting` |
| `--n-trials` | Number of Optuna optimization trials per model | `30` |
| `--forecast-days` | Number of days to forecast | `15` |
| `--start-date` | Start date for training data (YYYY-MM-DD) | 1 year ago |
| `--end-date` | End date for training data (YYYY-MM-DD) | today |

### Available Models

The following models can be selected via the `--models` argument:

**Sklearn-based models (default):**
| Model | Key | Description |
|-------|-----|-------------|
| Ridge Regression | `ridge` | L2 regularized linear regression |
| ElasticNet | `elasticnet` | Combined L1/L2 regularization |
| Random Forest | `random_forest` | Ensemble of decision trees |
| Gradient Boosting | `gradient_boosting` | Sequential ensemble method |

**Time series models:**
| Model | Key | Description |
|-------|-----|-------------|
| SARIMAX | `sarimax` | Seasonal ARIMA with exogenous variables (statsmodels) |
| Prophet | `prophet` | Facebook/Meta time series forecasting with seasonality |

**SARIMAX hyperparameters optimized:**
- `p`, `d`, `q` - Non-seasonal order (AR, differencing, MA)
- `P`, `D`, `Q`, `s` - Seasonal order

**Prophet hyperparameters optimized:**
- `changepoint_prior_scale` - Trend flexibility
- `seasonality_prior_scale` - Seasonality strength
- `seasonality_mode` - Additive or multiplicative
- `yearly_seasonality`, `weekly_seasonality` - Enable/disable seasonalities

The best model is selected based on test RMSE and used for generating forecasts.

### ML Architecture

The ML training pipeline is organized into modular components:

| Module | Location | Purpose |
|--------|----------|---------|
| Dataset | `src/dataset/` | Feature engineering, data preparation |
| Models | `src/statistical_models/` | Model classes with shared interface |
| Services | `src/services/` | Training, optimization, forecasting orchestration |

All models implement a common interface:
- `fit(X, y, features, **params)` - Train the model
- `predict(X)` - Make predictions
- `get_hyperparameter_space(trial)` - Define Optuna search space
- `get_feature_importances()` - Get feature importance scores

### Output

- **MLflow**: All experiments are tracked in MLflow (view at `http://localhost:5000`)
- **Database**: Predictions are saved to the `forecast_data` table with:
  - `predicted_price` - Always stored as USD price
  - `upper_predicted_price` / `lower_predicted_price` - Confidence bounds
  - `forecast_run_date` - Timestamp of the prediction run

## PDF Export

Export Jupyter notebooks to PDF with a custom cover page.

### Usage

```bash
# Using the export script
uv run python scripts/export_pdf.py notebooks/your_notebook.ipynb

# With custom output name
uv run python scripts/export_pdf.py notebooks/your_notebook.ipynb --output my_report

# Using makefile
make export-pdf
```

### Customization

- **Cover image**: Replace `templates/custom_report/assets/cover_page.jpg`
- **Template**: Modify `templates/custom_report/index.tex.j2`

### Requirements

PDF export requires LaTeX to be installed:

```bash
# Ubuntu/Debian
sudo apt install texlive-xetex texlive-fonts-recommended texlive-plain-generic
```

## Metabase

Metabase is included for data visualization and analytics.

### Accessing Metabase

After starting the services with `docker compose up -d`, access Metabase at:

- URL: http://localhost:3000
- First-time setup will prompt you to create an admin account

### Connecting to Coin Data

During Metabase setup, add a PostgreSQL database connection:

| Setting | Value |
|---------|-------|
| Host | `postgres` |
| Port | `5432` |
| Database | `cryptodb` |
| Username | Value from `.env` `POSTGRES_USER` |
| Password | Value from `.env` `POSTGRES_PASSWORD` |
