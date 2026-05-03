# =============================================================================
# INFRAESTRUCTURA
# =============================================================================

up:
	docker compose up --build -d postgres mlflow

up-core:
	docker compose up -d postgres mlflow metabase

down:
	docker compose down

logs:
	docker compose logs -f

# =============================================================================
# AIRFLOW — apache/airflow:2.8.1-python3.11 / localhost:8080
# Primera vez: make airflow-init  (inicializa BD + crea usuario admin)
# Arranque normal: make airflow-up
# Trigger manual: make airflow-trigger
# =============================================================================

AIRFLOW_CLI = docker compose exec airflow-scheduler /home/airflow/.local/bin/airflow

airflow-init:
	docker compose up airflow-init

airflow-up:
	docker compose up -d airflow-webserver airflow-scheduler

airflow-down:
	docker compose stop airflow-webserver airflow-scheduler

airflow-logs:
	docker compose logs -f airflow-webserver airflow-scheduler

airflow-trigger:
	$(AIRFLOW_CLI) dags trigger debit_ml_pipeline

airflow-status:
	$(AIRFLOW_CLI) dags list-runs --dag-id debit_ml_pipeline --output table

airflow-pause:
	$(AIRFLOW_CLI) dags pause debit_ml_pipeline

airflow-unpause:
	$(AIRFLOW_CLI) dags unpause debit_ml_pipeline

# =============================================================================
# CARGA DE DATOS — src/dataset/loader.py
# Params: DATALAKE_PATH, CHUNK_SIZE, TABLES, TRUNCATE
# TABLES opcionales: clientes excedentes gestiones moras pagos canales
# =============================================================================

load:
	uv run python -m src.dataset.loader \
		--datalake-path ./datalake \
		--chunk-size 5000 \
		--truncate

load-partial:
	uv run python -m src.dataset.loader \
		--datalake-path $(DATALAKE_PATH) \
		--chunk-size $(CHUNK_SIZE) \
		--table $(TABLES) \
		$(if $(TRUNCATE),--truncate,)

load-clientes:
	uv run python -m src.dataset.loader \
		--datalake-path ./datalake \
		--table clientes \
		--truncate

load-pagos:
	uv run python -m src.dataset.loader \
		--datalake-path ./datalake \
		--table pagos \
		--truncate

load-canales:
	uv run python -m src.dataset.loader \
		--datalake-path ./datalake \
		--table canales \
		--chunk-size 50 \
		--truncate

# =============================================================================
# MODELO ANALÍTICO — src/dataset/data_preparation.py
# Params: DATALAKE_PATH, OUTPUT_DIR
# =============================================================================

build-model:
	uv run python -m src.dataset.data_preparation \
		--datalake-path ./datalake \
		--output-dir ./data/artifacts

build-model-custom:
	uv run python -m src.dataset.data_preparation \
		--datalake-path $(DATALAKE_PATH) \
		--output-dir $(OUTPUT_DIR)

# =============================================================================
# FEATURE ENGINEERING — src/dataset/feature_engineering.py
# Params: INPUT, OUTPUT_DIR, SPLIT_STRATEGY (temporal|random),
#         RANDOM_STATE, TRAIN_SIZE, TEST_SIZE
# =============================================================================

features:
	uv run python -m src.dataset.feature_engineering \
		--input data/artifacts/analytical_model.parquet \
		--output-dir data/artifacts \
		--split-strategy temporal

features-random:
	uv run python -m src.dataset.feature_engineering \
		--input data/artifacts/analytical_model.parquet \
		--output-dir data/artifacts \
		--split-strategy random \
		--random-state 42 \
		--train-size 0.477 \
		--test-size 0.293

features-custom:
	uv run python -m src.dataset.feature_engineering \
		--input $(INPUT) \
		--output-dir $(OUTPUT_DIR) \
		--split-strategy $(or $(SPLIT_STRATEGY),temporal) \
		--random-state $(or $(RANDOM_STATE),42) \
		--train-size $(or $(TRAIN_SIZE),0.477) \
		--test-size $(or $(TEST_SIZE),0.293)

# =============================================================================
# SELECCIÓN DE FEATURES — src/dataset/feature_selection.py
# Params: TRAIN_PATH, OUTPUT_DIR, VAR_THRESHOLD, ANOVA_PERCENTILE,
#         ELASTICNET_L1, ELASTICNET_C
# =============================================================================

select-features:
	uv run python -m src.dataset.feature_selection \
		--train-path data/artifacts/train.parquet \
		--output-dir data/artifacts \
		--var-threshold 0.01 \
		--anova-percentile 80 \
		--elasticnet-l1 0.7 \
		--elasticnet-c 0.1

select-features-strict:
	uv run python -m src.dataset.feature_selection \
		--train-path data/artifacts/train.parquet \
		--output-dir data/artifacts \
		--var-threshold 0.05 \
		--anova-percentile 70 \
		--elasticnet-l1 1.0 \
		--elasticnet-c 0.05

select-features-custom:
	uv run python -m src.dataset.feature_selection \
		--train-path $(or $(TRAIN_PATH),data/artifacts/train.parquet) \
		--output-dir $(or $(OUTPUT_DIR),data/artifacts) \
		--var-threshold $(or $(VAR_THRESHOLD),0.01) \
		--anova-percentile $(or $(ANOVA_PERCENTILE),80) \
		--elasticnet-l1 $(or $(ELASTICNET_L1),0.7) \
		--elasticnet-c $(or $(ELASTICNET_C),0.1)

# =============================================================================
# ENTRENAMIENTO — deploy/train_debit_classifier.py
# Params: DATA_DIR, MLFLOW_URI, N_TRIALS, MODELS, SPLIT_STRATEGY
# MODELS opcionales: xgboost gradient_boosting logistic_regression random_forest
# =============================================================================

train:
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5 \
		--split-strategy temporal

train-all-models:
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5 \
		--models xgboost gradient_boosting logistic_regression random_forest \
		--split-strategy temporal

train-xgboost:
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 10 \
		--models xgboost \
		--split-strategy temporal

train-baseline:
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 0 \
		--models logistic_regression \
		--split-strategy temporal

train-random-split:
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5 \
		--split-strategy random

train-custom:
	uv run python deploy/train_debit_classifier.py \
		--data-dir $(or $(DATA_DIR),data/artifacts) \
		--mlflow-uri $(or $(MLFLOW_URI),http://localhost:5000) \
		--n-trials $(or $(N_TRIALS),5) \
		--models $(or $(MODELS),xgboost gradient_boosting logistic_regression) \
		--split-strategy $(or $(SPLIT_STRATEGY),temporal)

# =============================================================================
# PIPELINE COMPLETO — secuencia T1→T2→T3→T3.5→T4
# =============================================================================

pipeline:
	uv run python -m src.dataset.loader \
		--datalake-path ./datalake \
		--chunk-size 5000 \
		--truncate
	uv run python -m src.dataset.data_preparation \
		--datalake-path ./datalake \
		--output-dir ./data/artifacts
	uv run python -m src.dataset.feature_engineering \
		--input data/artifacts/analytical_model.parquet \
		--output-dir data/artifacts \
		--split-strategy temporal
	uv run python -m src.dataset.feature_selection \
		--train-path data/artifacts/train.parquet \
		--output-dir data/artifacts
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5

pipeline-from-features:
	uv run python -m src.dataset.feature_engineering \
		--input data/artifacts/analytical_model.parquet \
		--output-dir data/artifacts \
		--split-strategy temporal
	uv run python -m src.dataset.feature_selection \
		--train-path data/artifacts/train.parquet \
		--output-dir data/artifacts
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5

# =============================================================================
# DIAGNÓSTICO — Alerta 1: verificar separación perfecta
# =============================================================================

diag-random-split:
	uv run python -m src.dataset.feature_engineering \
		--input data/artifacts/analytical_model.parquet \
		--output-dir data/artifacts \
		--split-strategy random \
		--random-state 42
	uv run python deploy/train_debit_classifier.py \
		--data-dir data/artifacts \
		--mlflow-uri http://localhost:5000 \
		--n-trials 5 \
		--split-strategy random

scan-inf:
	uv run python scripts/scan_inf.py

# =============================================================================
# METABASE — deploy/metabase_provisioning.py (configurado por variables de entorno)
# Vars: METABASE_URL, METABASE_USER, METABASE_PASS,
#       DEBIT_DB_HOST, DEBIT_DB_NAME, DEBIT_DB_USER, DEBIT_DB_PASS, DEBIT_DB_PORT
# =============================================================================

export-model-metrics:
	uv run python deploy/export_model_metrics.py \
		--mlflow-uri http://localhost:5000 \
		--top-features 50

export-model-metrics-overwrite:
	uv run python deploy/export_model_metrics.py \
		--mlflow-uri http://localhost:5000 \
		--top-features 50 \
		--overwrite

provision-metabase:
	METABASE_URL=http://localhost:3000 \
	METABASE_USER=admin@bancolombia.com \
	METABASE_PASS=Debit2026!Bancolombia \
	DEBIT_DB_HOST=postgres \
	DEBIT_DB_NAME=debitdb \
	DEBIT_DB_USER=fredy \
	DEBIT_DB_PASS=password \
	DEBIT_DB_PORT=5432 \
	uv run python deploy/metabase_provisioning.py

provision-metabase-docker:
	METABASE_URL=http://metabase:3000 \
	METABASE_USER=admin@bancolombia.com \
	METABASE_PASS=admin123 \
	DEBIT_DB_HOST=postgres \
	DEBIT_DB_NAME=debitdb \
	DEBIT_DB_USER=debit \
	DEBIT_DB_PASS=debit \
	DEBIT_DB_PORT=5432 \
	uv run python deploy/metabase_provisioning.py

# =============================================================================
# MIGRACIONES — Alembic
# =============================================================================

migrate:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1

migrate-history:
	uv run alembic history --verbose

# =============================================================================
# AYUDA
# =============================================================================

help:
	@echo ""
	@echo "=== INFRAESTRUCTURA ==="
	@echo "  make up                   Levanta postgres + mlflow"
	@echo "  make up-core              postgres + mlflow + metabase"
	@echo "  make down                 Detiene todos los servicios"
	@echo ""
	@echo "=== AIRFLOW (localhost:8080 — admin/admin) ==="
	@echo "  make airflow-init         Primera vez: migra BD + crea usuario admin"
	@echo "  make airflow-up           Arranca webserver + scheduler"
	@echo "  make airflow-down         Detiene webserver + scheduler"
	@echo "  make airflow-logs         Logs en tiempo real"
	@echo "  make airflow-trigger      Dispara el DAG debit_ml_pipeline manualmente"
	@echo "  make airflow-status       Lista runs del DAG"
	@echo "  make airflow-pause        Pausa el DAG (evita ejecución @monthly)"
	@echo "  make airflow-unpause      Activa el DAG"
	@echo ""
	@echo "=== CARGA DE DATOS ==="
	@echo "  make load                 Carga todos los CSVs (default)"
	@echo "  make load-clientes        Solo tabla clientes"
	@echo "  make load-pagos           Solo tabla pagos"
	@echo "  make load-canales         Canales (chunk-size 50, streaming)"
	@echo "  make load-partial DATALAKE_PATH=./datalake CHUNK_SIZE=5000 TABLES='pagos gestiones' TRUNCATE=1"
	@echo ""
	@echo "=== MODELO ANALÍTICO ==="
	@echo "  make build-model          LEFT JOIN → analytical_model.parquet"
	@echo "  make build-model-custom DATALAKE_PATH=./datalake OUTPUT_DIR=./data/artifacts"
	@echo ""
	@echo "=== FEATURE ENGINEERING ==="
	@echo "  make features             Split temporal (producción)"
	@echo "  make features-random      Split aleatorio (diagnóstico)"
	@echo "  make features-custom INPUT=... OUTPUT_DIR=... [SPLIT_STRATEGY=random] [RANDOM_STATE=42] [TRAIN_SIZE=0.477] [TEST_SIZE=0.293]"
	@echo ""
	@echo "=== SELECCIÓN DE FEATURES ==="
	@echo "  make select-features      Defaults: var=0.01, anova=80, l1=0.7, c=0.1"
	@echo "  make select-features-strict  Más agresivo: var=0.05, anova=70, l1=1.0, c=0.05"
	@echo "  make select-features-custom [TRAIN_PATH=...] [VAR_THRESHOLD=0.01] [ANOVA_PERCENTILE=80] [ELASTICNET_L1=0.7] [ELASTICNET_C=0.1]"
	@echo ""
	@echo "=== ENTRENAMIENTO ==="
	@echo "  make train                Default: xgboost + gradient_boosting + logistic_regression, 5 trials"
	@echo "  make train-all-models     Incluye random_forest"
	@echo "  make train-xgboost        Solo XGBoost, 10 trials"
	@echo "  make train-baseline       Solo LogisticRegression, 0 trials"
	@echo "  make train-random-split   Split aleatorio (diagnóstico Alerta 1)"
	@echo "  make train-custom [DATA_DIR=...] [MLFLOW_URI=...] [N_TRIALS=5] [MODELS='xgboost ...'] [SPLIT_STRATEGY=temporal]"
	@echo ""
	@echo "=== PIPELINE COMPLETO ==="
	@echo "  make pipeline             T1→T2→T3→T3.5→T4 desde cero"
	@echo "  make pipeline-from-features  T3→T3.5→T4 (reutiliza analytical_model.parquet)"
	@echo ""
	@echo "=== DIAGNÓSTICO ==="
	@echo "  make diag-random-split    Detecta separación perfecta (Alerta 1)"
	@echo "  make scan-inf             Escanea inf en train/test/oot.parquet"
	@echo ""
	@echo "=== METABASE ==="
	@echo "  make provision-metabase        Provisiona desde localhost"
	@echo "  make provision-metabase-docker Provisiona desde red Docker interna"
	@echo ""
	@echo "=== MIGRACIONES ==="
	@echo "  make migrate              alembic upgrade head"
	@echo "  make migrate-down         alembic downgrade -1"
	@echo "  make migrate-history      historial de migraciones"
	@echo ""
