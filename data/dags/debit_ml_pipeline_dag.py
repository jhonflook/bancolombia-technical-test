"""DAG: Pipeline ML de débitos recurrentes — Bancolombia.

Orquesta el pipeline completo de clasificación binaria:
  1. setup_artifacts_dir     → Crea directorio de artefactos en el host
  2. load_data               → Carga 6 CSVs a PostgreSQL (debit_* tables)
  3. build_analytical_model  → Construye modelo analítico unificado (.parquet)
  4. build_features          → Feature engineering + split Train/Test/OOT (S7)
  5. train_model             → Entrena XGBoost + evalúa (AUC, KS) + registra en MLflow
  6. provision_metabase      → Crea/actualiza dashboard Metabase vía API REST

Cada tarea corre en el contenedor debit-ml:latest sobre la red debit_network,
lo que garantiza acceso a postgres y mlflow sin exponer puertos al host.

Supuestos de infraestructura:
  - IMAGE:         debit-ml:latest  (construida con `docker compose build`)
  - NETWORK:       debit_network    (definida en docker-compose.yml)
  - DATALAKE:      montado desde ${PROJECT_ROOT}/datalake en /app/datalake
  - ARTIFACTS:     montado desde ${PROJECT_ROOT}/data/artifacts en /app/data/artifacts
  - POSTGRES_HOST: postgres  (nombre del servicio en debit_network)
  - MLFLOW_URI:    http://mlflow:5000 (nombre del servicio en debit_network)

Ejecución manual:
    airflow dags trigger debit_ml_pipeline
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ─── Configuración de infraestructura ────────────────────────────────────────

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/jhonflook/Documents/analitico lll - Bancolombia/approach2",
)
IMAGE = "debit-ml:latest"
NETWORK = "debit_network"
DOCKER_URL = "unix://var/run/docker.sock"

# Variables de entorno inyectadas en cada contenedor Docker
CONTAINER_ENV = {
    "POSTGRES_HOST": "postgres",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}

# Volúmenes compartidos entre tareas
DATALAKE_MOUNT = Mount(
    source=f"{PROJECT_ROOT}/datalake",
    target="/app/datalake",
    type="bind",
    read_only=True,
)
ARTIFACTS_MOUNT = Mount(
    source=f"{PROJECT_ROOT}/data/artifacts",
    target="/app/data/artifacts",
    type="bind",
    read_only=False,
)

# ─── DAG ─────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "depends_on_past": False,
}

with DAG(
    dag_id="debit_ml_pipeline",
    description=(
        "Pipeline ML débitos recurrentes: "
        "carga CSV → modelo analítico → features → XGBoost + MLflow"
    ),
    schedule_interval="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["debit", "ml", "bancolombia"],
    doc_md=__doc__,
) as dag:

    # ── T0: Crear directorio de artefactos en el host ─────────────────────
    # Garantiza que el directorio existe con los permisos correctos antes de que
    # Docker lo monte como volumen (si Docker lo crea, lo hace como root).
    setup_artifacts_dir = BashOperator(
        task_id="setup_artifacts_dir",
        bash_command=(
            f"mkdir -p '{PROJECT_ROOT}/data/artifacts' && "
            f"echo 'Artifacts dir OK: {PROJECT_ROOT}/data/artifacts'"
        ),
        doc_md="Crea data/artifacts/ en el host antes del primer montaje Docker.",
    )

    # ── T1: Carga CSV → PostgreSQL ────────────────────────────────────────
    # Trunca las tablas debit_* e inserta los 6 CSVs por chunks de 5000 filas.
    # --truncate garantiza idempotencia en re-ejecuciones mensuales del DAG.
    # El entrypoint.sh corre `alembic upgrade head` antes de invocar el loader.
    load_data = DockerOperator(
        task_id="load_data",
        image=IMAGE,
        command=(
            "uv run python -m src.dataset.loader "
            "--datalake-path /app/datalake "
            "--chunk-size 5000 "
            "--truncate"
        ),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[DATALAKE_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md=(
            "Carga los 6 CSVs a PostgreSQL (debit_clients, debit_excedentes, "
            "debit_gestiones, debit_moras, debit_pagos, debit_canales). "
            "Aplica S4 (canales ausentes → 0 en JSONB), S5 (nulos excedentes → NULL), "
            "S11 (24 grupos canales conflictivos → excluidos)."
        ),
    )

    # ── T2: Modelo analítico unificado ────────────────────────────────────
    # Lee los 6 CSVs directamente (más eficiente que leer desde BD para ML),
    # realiza LEFT JOINs, imputa nulos y persiste analytical_model.parquet.
    # Lee desde datalake/ y escribe en data/artifacts/ — ambos montados.
    build_analytical_model = DockerOperator(
        task_id="build_analytical_model",
        image=IMAGE,
        command=(
            "uv run python -m src.dataset.data_preparation "
            "--datalake-path /app/datalake "
            "--output-dir /app/data/artifacts"
        ),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[DATALAKE_MOUNT, ARTIFACTS_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md=(
            "Construye el modelo analítico unificado (granularidad: "
            "num_doc × obl17 × f_analisis) mediante LEFT JOIN de las 6 fuentes. "
            "Aplica S4, S5. Valida unicidad de clave primaria (S1). "
            "Salida: data/artifacts/analytical_model.parquet (~46k filas)."
        ),
    )

    # ── T3: Feature engineering + split temporal ──────────────────────────
    # Lee analytical_model.parquet, construye features derivadas (exclusividad
    # de canal, recurrencia de débito, ratios de gestiones, coeficientes de mora,
    # indicadores de excedente) y guarda los splits temporales definidos en S7.
    build_features = DockerOperator(
        task_id="build_features",
        image=IMAGE,
        command=(
            "uv run python -m src.dataset.feature_engineering "
            "--input /app/data/artifacts/analytical_model.parquet "
            "--output-dir /app/data/artifacts"
        ),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[ARTIFACTS_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md=(
            "Aplica feature engineering y split Train/Test/OOT (S7): "
            "Train 2024-07 → 2025-02, Test 2025-03 → 2025-07, OOT 2025-08 → 2025-11. "
            "Filtros S10: varianza cero y sparsity > 99%%. "
            "Salidas: train.parquet, test.parquet, oot.parquet, feature_cols.json."
        ),
    )

    # ── T4: Entrenamiento + evaluación + MLflow ───────────────────────────
    # Entrena XGBoost con scale_pos_weight = n_neg/n_pos para el desbalance 3.7:1.
    # Evalúa en las 3 particiones con AUC-ROC, KS, Precision/Recall, AUC-PR.
    # Registra parámetros, métricas, feature importance y modelo en MLflow.
    train_model = DockerOperator(
        task_id="train_model",
        image=IMAGE,
        command=(
            "uv run python -m src.services.training "
            "--data-dir /app/data/artifacts "
            "--mlflow-uri http://mlflow:5000"
        ),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[ARTIFACTS_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        execution_timeout=timedelta(hours=2),
        doc_md=(
            "Entrena XGBoost binario con corrección por desbalance (scale_pos_weight, S8). "
            "Métricas: AUC-ROC (primaria, S8), KS, Precision/Recall@0.5 y @umbral KS, AUC-PR. "
            "Artefactos: model_xgboost.pkl, feature_importance.csv → MLflow run."
        ),
    )

    # ── T5: Provisionamiento del tablero Metabase ─────────────────────────
    # Configura Metabase vía API REST: conexión a debitdb, 10 questions
    # (usando vistas v_debit_*) y dashboard ensamblado.
    # Es idempotente: reutiliza cards y dashboard si ya existen.
    provision_metabase = DockerOperator(
        task_id="provision_metabase",
        image=IMAGE,
        command="uv run python /app/deploy/metabase_provisioning.py",
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment={
            **CONTAINER_ENV,
            "METABASE_URL":  "http://metabase:3000",
            "METABASE_USER": "admin@bancolombia.com",
            "METABASE_PASS": "admin123",
            "DEBIT_DB_HOST": "postgres",
            "DEBIT_DB_NAME": "debitdb",
            "DEBIT_DB_USER": "debit",
            "DEBIT_DB_PASS": "debit",
        },
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md=(
            "Provisiona el tablero Metabase con 10 preguntas analíticas de negocio "
            "basadas en las vistas v_debit_* de PostgreSQL. "
            "Segmentos: A=Automatizar, B=Monitorear, C=Cobranza suave, D=Cobranza intensiva."
        ),
    )

    # ─── Dependencias (pipeline lineal) ──────────────────────────────────
    (
        setup_artifacts_dir
        >> load_data
        >> build_analytical_model
        >> build_features
        >> train_model
        >> provision_metabase
    )
