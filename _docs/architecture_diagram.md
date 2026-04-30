# Project Architecture Diagram

## High-Level Architecture

```mermaid
flowchart TB
    subgraph External["External Services"]
        CoinGecko["CoinGecko API"]
    end

    subgraph Docker["Docker Infrastructure"]
        subgraph Airflow["Apache Airflow"]
            Webserver["Airflow Webserver<br/>:8080"]
            Scheduler["Airflow Scheduler"]
            Init["Airflow Init"]
        end

        subgraph Storage["Data Storage"]
            Postgres["PostgreSQL :5432"]
            subgraph Databases["Databases"]
                AirflowDB["airflow"]
                CryptoDB["cryptodb"]
                MetabaseDB["metabaseappdb"]
                MLflowDB["mlflowdb"]
            end
        end

        CoinFetcher["Coin-Fetcher<br/>Container"]
        Metabase["Metabase<br/>:3000"]
        MLflowServer["MLflow Server<br/>:5000"]
    end

    subgraph DAGs["Airflow DAGs"]
        FetcherDAG["coin_fetcher_daily<br/>3am UTC Daily"]
        TrainDAG["coin_train_forecast_pct<br/>3am UTC Weekly"]
        StatsDAG["coin_statistical_analysis<br/>4am UTC Daily"]
    end

    CoinGecko --> FetcherDAG
    FetcherDAG --> CoinFetcher
    CoinFetcher --> CryptoDB
    TrainDAG --> CoinFetcher
    StatsDAG --> CryptoDB

    Scheduler --> DAGs
    Webserver --> Scheduler
    Init --> Webserver

    Postgres --> Databases
    MLflowServer --> MLflowDB
    Metabase --> CryptoDB
```

## Data Pipeline Flow

```mermaid
flowchart LR
    subgraph Acquisition["Data Acquisition"]
        API["CoinGecko API"]
        Fetcher["coin_fetcher.py"]
        JSON["JSON Files<br/>data/queries/"]
    end

    subgraph Storage["Storage Layer"]
        CoinData["coin_data<br/>Table"]
        MonthlyAgg["coin_monthly_aggregation<br/>Table"]
    end

    subgraph Processing["Statistical Processing"]
        StatsDAG["coin_statistical_analysis<br/>DAG"]
        ConsecDrop["_statistics_consecutive_drop_analysis"]
        MonthlyAvg["_statistics_monthly_average_price"]
    end

    subgraph ML["ML Pipeline"]
        Training["train_forecast_models_pct.py"]
        Predictions["forecast_data<br/>Table"]
    end

    API --> Fetcher
    Fetcher --> JSON
    Fetcher --> CoinData
    Fetcher --> MonthlyAgg

    CoinData --> StatsDAG
    StatsDAG --> ConsecDrop
    StatsDAG --> MonthlyAvg

    CoinData --> Training
    Training --> Predictions
```

## ML Training Architecture

```mermaid
flowchart TB
    subgraph EntryPoint["Entry Point"]
        CLI["train_forecast_models_pct.py"]
    end

    subgraph Services["Services Layer"]
        TrainingSvc["TrainingService"]
        OptSvc["HyperparameterOptimizer"]
        ForecastSvc["ForecastingService"]
    end

    subgraph Dataset["Dataset Module"]
        FeatureEng["Feature Engineering"]
        DataPrep["Data Preparation"]
        FeatureConfig["Feature Config"]
    end

    subgraph Models["Statistical Models"]
        subgraph Sklearn["Sklearn-Based"]
            Ridge["RidgeModel"]
            ElasticNet["ElasticNetModel"]
            RandomForest["RandomForestModel"]
            GradientBoosting["GradientBoostingModel"]
        end
        subgraph TimeSeries["Time Series"]
            SARIMAX["SARIMAXModel"]
            Prophet["ProphetModel"]
        end
    end

    subgraph Tracking["Experiment Tracking"]
        MLflow["MLflow"]
        Optuna["Optuna"]
    end

    subgraph Output["Output"]
        ForecastDB["forecast_data<br/>Table"]
    end

    CLI --> TrainingSvc
    TrainingSvc --> OptSvc
    TrainingSvc --> ForecastSvc

    TrainingSvc --> Dataset
    Dataset --> Models

    OptSvc --> Optuna
    TrainingSvc --> MLflow

    ForecastSvc --> ForecastDB
```

## Docker Services Interaction

```mermaid
flowchart TB
    subgraph Network["coin_network"]
        subgraph Core["Core Services"]
            Postgres["PostgreSQL<br/>:5432"]
        end

        subgraph Airflow["Airflow Stack"]
            Init["airflow-init"]
            Web["airflow-webserver<br/>:8080"]
            Sched["airflow-scheduler"]
        end

        subgraph Apps["Application Services"]
            Fetcher["coin-fetcher"]
            MLflow["mlflow<br/>:5000"]
            Metabase["metabase<br/>:3000"]
        end
    end

    subgraph Volumes["Volumes"]
        DAGsVol["./data/dags"]
        QueriesVol["./data/queries"]
        DBVol["./data/database"]
        MLflowVol["./data/mlflow"]
    end

    Postgres --> Init
    Init --> Web
    Init --> Sched
    Postgres --> Fetcher
    Postgres --> MLflow
    Postgres --> Metabase

    DAGsVol --> Web
    DAGsVol --> Sched
    QueriesVol --> Web
    QueriesVol --> Sched
    DBVol --> Postgres
    MLflowVol --> MLflow
```

## Database Schema

```mermaid
erDiagram
    coin_data {
        int id PK
        varchar coin_id
        float price_usd
        date date
        jsonb json_response
    }

    coin_monthly_aggregation {
        int id PK
        varchar coin_id
        int year
        int month
        float price_max
        float price_min
    }

    forecast_data {
        int id PK
        varchar coin_id
        date date
        varchar forecast_id FK
        float predicted_price
        float upper_predicted_price
        float lower_predicted_price
        timestamp forecast_run_date
    }

    forecast_metadata {
        int id PK
        varchar forecast_id UK
        varchar coin_id
        varchar model_type
        varchar target_type
        float mape
        float rmse
        float r2
        float mae
        int n_train_samples
        int n_test_samples
        int n_trials
        int n_features
        varchar selected_features
        int forecast_days
        date train_start_date
        date train_end_date
        timestamp forecast_run_date
        varchar tag
        varchar mlflow_run_id
    }

    _statistics_consecutive_drop_analysis {
        varchar coin_id PK
        int statistical_events
        numeric avg_drop_length_days
        numeric avg_price_increase_usd
        numeric avg_pct_increase
        numeric current_market_cap_usd
        date market_cap_date
    }

    _statistics_monthly_average_price {
        int id PK
        varchar coin_id
        int year
        int month
        numeric avg_price_usd
        numeric min_price_usd
        numeric max_price_usd
        int data_points
    }

    coin_data ||--o{ coin_monthly_aggregation : "aggregates"
    coin_data ||--o{ forecast_data : "predicts"
    coin_data ||--o{ _statistics_consecutive_drop_analysis : "analyzes"
    coin_data ||--o{ _statistics_monthly_average_price : "summarizes"
    forecast_metadata ||--o{ forecast_data : "describes"
```

## Project Structure

```mermaid
flowchart TB
    subgraph Root["Project Root"]
        subgraph Deploy["deploy/"]
            CoinFetcher["coin_fetcher.py<br/>Data Acquisition CLI"]
            TrainModels["train_forecast_models_pct.py<br/>ML Training Entry Point"]
            Dockerfile["Dockerfile"]
        end

        subgraph Data["data/"]
            DAGs["dags/<br/>Airflow DAG definitions"]
            Queries["queries/<br/>Cached JSON responses"]
        end

        subgraph Src["src/"]
            Database["database/<br/>Connections & CRUD"]
            DatasetMod["dataset/<br/>Feature Engineering"]
            Models["models/<br/>SQLModel Tables"]
            ServicesMod["services/<br/>Training, Optimization, Forecasting"]
            StatModels["statistical_models/<br/>ML Model Implementations"]
            Settings["settings.py<br/>Configuration"]
        end

        subgraph Config["Configuration"]
            DockerCompose["docker-compose.yml"]
            Env[".env"]
            Pyproject["pyproject.toml"]
        end
    end

    Deploy --> Src
    DAGs --> Deploy
    Src --> Database
```

## Airflow DAG Dependencies

```mermaid
flowchart LR
    subgraph Daily["Daily Schedule (3am UTC)"]
        FetchBTC["fetch_bitcoin"]
        FetchETH["fetch_ethereum"]
        FetchADA["fetch_cardano"]
    end

    subgraph Stats["Daily Schedule (4am UTC)"]
        CreateConsec["create_consecutive_drop_procedure"]
        ExecConsec["execute_consecutive_drop_analysis"]
        CreateMonthly["create_monthly_avg_procedure"]
        ExecMonthly["execute_monthly_avg_analysis"]
    end

    subgraph Weekly["Weekly Schedule (3am UTC Monday)"]
        TrainBTC["train_bitcoin"]
        TrainETH["train_ethereum"]
        TrainADA["train_cardano"]
    end

    Daily --> Stats
    CreateConsec --> ExecConsec
    CreateMonthly --> ExecMonthly
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | Apache Airflow 2.8.1 | DAG scheduling and execution |
| Database | PostgreSQL 17 | Data persistence |
| ML Tracking | MLflow | Experiment tracking |
| Optimization | Optuna | Hyperparameter tuning |
| Visualization | Metabase | Business intelligence |
| Containerization | Docker Compose | Service orchestration |
| Package Manager | UV | Python dependency management |
| ML Models | scikit-learn, statsmodels, Prophet | Prediction models |
