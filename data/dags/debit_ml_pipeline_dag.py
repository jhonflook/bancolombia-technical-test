"""DAG: Pipeline ML de débitos recurrentes — Bancolombia.

Cada tarea es controlable individualmente mediante parámetros del DAG.
Los defaults se leen del .env (disponible en los contenedores Airflow via
`env_file`); se pueden sobrescribir al disparar el DAG desde la UI o CLI.

Parámetros (boolean) configurables en .env:
  RUN_SETUP           → T0: crear directorio de artefactos
  RUN_LOAD_DATA       → T1: carga CSV → PostgreSQL
  RUN_BUILD_MODEL     → T2: modelo analítico unificado
  RUN_BUILD_FEATURES  → T3: feature engineering + split Train/Test/OOT
  RUN_SELECT_FEATURES → T3.5: selección supervisada de features
  RUN_TRAIN           → T4: entrenamiento + MLflow
  RUN_METABASE        → T5: provisionamiento tablero Metabase

Ejemplo — re-entrenar sin recargar datos ni reconstruir features:
    airflow dags trigger debit_ml_pipeline --conf '{
        "run_load_data": false,
        "run_build_model": false,
        "run_build_features": false,
        "run_select_features": false
    }'

Diseño de gates:
  Cada ShortCircuitOperator (gate_*) actúa de "interruptor" para su tarea:
  - ignore_downstream_trigger_rules=False → solo salta la tarea inmediata.
  - TriggerRule.ALL_DONE en los gates siguientes → el pipeline continúa
    aunque la tarea anterior haya sido saltada.

  gate_T0 → T0 → gate_T1 → T1 → gate_T2 → T2 → ... → gate_T5 → T5

Supuestos de infraestructura:
  - IMAGE:         debit-ml:latest  (construida con `docker compose build`)
  - NETWORK:       debit_network    (definida en docker-compose.yml)
  - DATALAKE:      montado desde ${PROJECT_ROOT}/datalake en /app/datalake
  - ARTIFACTS:     montado desde ${PROJECT_ROOT}/data/artifacts en /app/data/artifacts
  - POSTGRES_HOST: postgres  (nombre del servicio en debit_network)
  - MLFLOW_URI:    http://mlflow:5000 (nombre del servicio en debit_network)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from docker.types import Mount

# ─── Configuración de infraestructura ────────────────────────────────────────

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/jhonflook/Documents/projects/bancolombia-technical-test",
)
IMAGE = "debit-ml:latest"
NETWORK = "debit_network"
DOCKER_URL = "unix://var/run/docker.sock"

CONTAINER_ENV = {
    "POSTGRES_HOST": "postgres",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _env_bool(var: str, default: bool = True) -> bool:
    """Lee una variable de entorno como booleano (1/true/yes → True)."""
    return os.getenv(var, str(default)).lower().strip() in ("1", "true", "yes")


def _gate_callable(param_name: str):
    """Devuelve un callable para ShortCircuitOperator que lee el param del DAG."""
    def _check(**context):
        return bool(context["params"][param_name])
    return _check


# ─── Parámetros del DAG (defaults desde .env) ────────────────────────────────

DAG_PARAMS = {
    "run_setup": Param(
        default=_env_bool("RUN_SETUP"),
        type="boolean",
        description="T0: crear directorio de artefactos",
    ),
    "run_load_data": Param(
        default=_env_bool("RUN_LOAD_DATA"),
        type="boolean",
        description="T1: cargar CSV → PostgreSQL",
    ),
    "run_build_model": Param(
        default=_env_bool("RUN_BUILD_MODEL"),
        type="boolean",
        description="T2: modelo analítico unificado (LEFT JOIN)",
    ),
    "run_build_features": Param(
        default=_env_bool("RUN_BUILD_FEATURES"),
        type="boolean",
        description="T3: feature engineering + split Train/Test/OOT",
    ),
    "run_select_features": Param(
        default=_env_bool("RUN_SELECT_FEATURES"),
        type="boolean",
        description="T3.5: selección supervisada de features (varianza→ANOVA→ElasticNet)",
    ),
    "run_train": Param(
        default=_env_bool("RUN_TRAIN"),
        type="boolean",
        description="T4: entrenamiento + evaluación + registro MLflow",
    ),
    "run_metabase": Param(
        default=_env_bool("RUN_METABASE"),
        type="boolean",
        description="T5: provisionamiento tablero Metabase",
    ),
}

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
        "carga CSV → modelo analítico → features → clasificadores + MLflow"
    ),
    schedule_interval="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    params=DAG_PARAMS,
    tags=["debit", "ml", "bancolombia"],
    doc_md=__doc__,
) as dag:

    # ══════════════════════════════════════════════════════════════════════
    # GATES — ShortCircuitOperator por tarea
    # ignore_downstream_trigger_rules=False → solo omite la tarea inmediata.
    # TriggerRule.ALL_DONE (gates T1 en adelante) → continúa aunque la
    # tarea anterior haya sido saltada.
    # ══════════════════════════════════════════════════════════════════════

    gate_setup = ShortCircuitOperator(
        task_id="gate_setup",
        python_callable=_gate_callable("run_setup"),
        ignore_downstream_trigger_rules=False,
        doc_md="Ejecuta T0 si `run_setup=true` (default: RUN_SETUP en .env).",
    )

    gate_load_data = ShortCircuitOperator(
        task_id="gate_load_data",
        python_callable=_gate_callable("run_load_data"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T1 si `run_load_data=true` (default: RUN_LOAD_DATA en .env).",
    )

    gate_build_model = ShortCircuitOperator(
        task_id="gate_build_model",
        python_callable=_gate_callable("run_build_model"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T2 si `run_build_model=true` (default: RUN_BUILD_MODEL en .env).",
    )

    gate_build_features = ShortCircuitOperator(
        task_id="gate_build_features",
        python_callable=_gate_callable("run_build_features"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T3 si `run_build_features=true` (default: RUN_BUILD_FEATURES en .env).",
    )

    gate_select_features = ShortCircuitOperator(
        task_id="gate_select_features",
        python_callable=_gate_callable("run_select_features"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T3.5 si `run_select_features=true` (default: RUN_SELECT_FEATURES en .env).",
    )

    gate_train = ShortCircuitOperator(
        task_id="gate_train",
        python_callable=_gate_callable("run_train"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T4 si `run_train=true` (default: RUN_TRAIN en .env).",
    )

    gate_metabase = ShortCircuitOperator(
        task_id="gate_metabase",
        python_callable=_gate_callable("run_metabase"),
        ignore_downstream_trigger_rules=False,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Ejecuta T5 si `run_metabase=true` (default: RUN_METABASE en .env).",
    )

    # ══════════════════════════════════════════════════════════════════════
    # TAREAS — DockerOperator
    # ══════════════════════════════════════════════════════════════════════

    # ── T0: Crear directorio de artefactos ───────────────────────────────
    setup_artifacts_dir = DockerOperator(
        task_id="setup_artifacts_dir",
        image=IMAGE,
        command="mkdir -p /app/data/artifacts && echo 'Artifacts dir OK'",
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[ARTIFACTS_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md="Crea /app/data/artifacts (montado desde PROJECT_ROOT/data/artifacts en el host).",
    )

    # ── T1: Carga CSV → PostgreSQL ────────────────────────────────────────
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
            "Aplica S4 (canales ausentes → 0), S5 (nulos excedentes → NULL), "
            "S11 (88 claves canales conflictivas → excluidas)."
        ),
    )

    # ── T2: Modelo analítico unificado ────────────────────────────────────
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
            "Guarda todas las features candidatas (~240 cols) sin filtrar. "
            "Salidas: train.parquet, test.parquet, oot.parquet."
        ),
    )

    # ── T3.5: Selección supervisada de features ───────────────────────────
    select_features = DockerOperator(
        task_id="select_features",
        image=IMAGE,
        command=(
            "uv run python -m src.dataset.feature_selection "
            "--train-path /app/data/artifacts/train.parquet "
            "--output-dir /app/data/artifacts"
        ),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        environment=CONTAINER_ENV,
        mounts=[ARTIFACTS_MOUNT],
        auto_remove="force",
        mount_tmp_dir=False,
        doc_md=(
            "Selección supervisada de features (fit solo en train, anti-leakage): "
            "1. Filtro varianza cero + sparsity >= 99%%. "
            "2. ANOVA F-test: top 80%% por F-score univariado. "
            "3. ElasticNet (l1_ratio=0.7, C=0.1): elimina features con coef=0. "
            "Salida: data/artifacts/feature_cols.json."
        ),
    )

    # ── T4: Entrenamiento + evaluación + MLflow ───────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════
    # DEPENDENCIAS
    # Cada gate activa o salta su tarea; el siguiente gate arranca siempre
    # (ALL_DONE) sin importar si la tarea fue ejecutada o saltada.
    # ══════════════════════════════════════════════════════════════════════
    (
        gate_setup >> setup_artifacts_dir
        >> gate_load_data >> load_data
        >> gate_build_model >> build_analytical_model
        >> gate_build_features >> build_features
        >> gate_select_features >> select_features
        >> gate_train >> train_model
        >> gate_metabase >> provision_metabase
    )
