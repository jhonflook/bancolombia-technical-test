"""create pipeline metrics table

Revision ID: f4g5h6i7j8k9
Revises: e3f4a5b6c7d8
Create Date: 2026-05-03
"""

from alembic import op
import sqlalchemy as sa

revision = "f4g5h6i7j8k9"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None

_VIEWS = [
    "v_debit_pipeline_load_summary",
    "v_debit_pipeline_feature_funnel",
    "v_debit_pipeline_splits",
]


def upgrade() -> None:
    op.create_table(
        "debit_pipeline_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id",         sa.String(64),  nullable=False),
        sa.Column("run_date",       sa.DateTime(timezone=True), nullable=False),
        sa.Column("split_strategy", sa.String(16),  nullable=False),
        sa.Column("stage",          sa.String(32),  nullable=False),
        sa.Column("source_name",    sa.String(64),  nullable=False),
        sa.Column("metric_name",    sa.String(128), nullable=False),
        sa.Column("metric_value",   sa.Float,       nullable=False),
        sa.UniqueConstraint(
            "run_id", "stage", "source_name", "metric_name",
            name="uq_pipeline_metrics_key",
        ),
        schema="public",
    )
    op.create_index("ix_pipeline_metrics_run_id", "debit_pipeline_metrics", ["run_id"])
    op.create_index("ix_pipeline_metrics_stage",  "debit_pipeline_metrics", ["stage"])
    op.create_index("ix_pipeline_metrics_split",  "debit_pipeline_metrics", ["split_strategy"])
    op.create_index("ix_pipeline_metrics_date",   "debit_pipeline_metrics", ["run_date"])

    # ── Vista 1: retención de datos en carga ─────────────────────────────────
    op.execute("""
        CREATE OR REPLACE VIEW v_debit_pipeline_load_summary AS
        SELECT
            run_id,
            split_strategy,
            source_name                                                                   AS fuente,
            MAX(CASE WHEN metric_name = 'rows_raw'            THEN metric_value END)     AS rows_bruto,
            MAX(CASE WHEN metric_name = 'rows_after_dedup'    THEN metric_value END)     AS rows_tras_dedup,
            MAX(CASE WHEN metric_name = 'rows_excluded'       THEN metric_value END)     AS rows_excluidos,
            MAX(CASE WHEN metric_name = 'rows_loaded'         THEN metric_value END)     AS rows_cargados,
            MAX(CASE WHEN metric_name = 'dup_exact_count'     THEN metric_value END)     AS dup_exactos,
            MAX(CASE WHEN metric_name = 'dup_conflict_count'  THEN metric_value END)     AS dup_conflicto,
            MAX(CASE WHEN metric_name = 'cols_count'          THEN metric_value END)     AS num_columnas,
            ROUND(
                (
                    100.0 * MAX(CASE WHEN metric_name = 'rows_loaded' THEN metric_value END)
                    / NULLIF(MAX(CASE WHEN metric_name = 'rows_raw' THEN metric_value END), 0)
                )::numeric
            , 2)                                                                          AS retencion_pct
        FROM debit_pipeline_metrics
        WHERE stage = 'loader'
        GROUP BY run_id, split_strategy, source_name
        ORDER BY source_name
    """)

    # ── Vista 2: funnel de selección de features ──────────────────────────────
    op.execute("""
        CREATE OR REPLACE VIEW v_debit_pipeline_feature_funnel AS
        SELECT
            run_id,
            split_strategy,
            MAX(CASE WHEN metric_name = 'cols_initial'            THEN metric_value END) AS cols_inicial,
            MAX(CASE WHEN metric_name = 'cols_after_variance'     THEN metric_value END) AS cols_tras_varianza,
            MAX(CASE WHEN metric_name = 'cols_after_anova'        THEN metric_value END) AS cols_tras_anova,
            MAX(CASE WHEN metric_name = 'cols_final'              THEN metric_value END) AS cols_final,
            MAX(CASE WHEN metric_name = 'removed_variance_var'    THEN metric_value END) AS removed_baja_var,
            MAX(CASE WHEN metric_name = 'removed_variance_sparse' THEN metric_value END) AS removed_sparse,
            MAX(CASE WHEN metric_name = 'cols_removed_anova'      THEN metric_value END) AS removed_anova,
            MAX(CASE WHEN metric_name = 'cols_removed_elasticnet' THEN metric_value END) AS removed_elasticnet,
            MAX(CASE WHEN metric_name = 'var_threshold'           THEN metric_value END) AS var_threshold,
            MAX(CASE WHEN metric_name = 'anova_percentile'        THEN metric_value END) AS anova_percentile,
            MAX(CASE WHEN metric_name = 'elasticnet_l1'           THEN metric_value END) AS elasticnet_l1,
            MAX(CASE WHEN metric_name = 'elasticnet_c'            THEN metric_value END) AS elasticnet_c,
            MAX(CASE WHEN metric_name = 'time_sec'                THEN metric_value END) AS tiempo_sec
        FROM debit_pipeline_metrics
        WHERE stage = 'feature_selection'
          AND source_name = 'overall'
        GROUP BY run_id, split_strategy
    """)

    # ── Vista 3: resumen de particiones Train/Test/OOT ────────────────────────
    op.execute("""
        CREATE OR REPLACE VIEW v_debit_pipeline_splits AS
        SELECT
            run_id,
            split_strategy,
            source_name                                                                        AS particion,
            MAX(CASE WHEN metric_name = 'rows'           THEN metric_value END)::int          AS filas,
            MAX(CASE WHEN metric_name = 'cols_candidate' THEN metric_value END)::int          AS features_candidatas,
            MAX(CASE WHEN metric_name = 'cols_selected'  THEN metric_value END)::int          AS features_seleccionadas,
            ROUND(
                MAX(CASE WHEN metric_name = 'target_rate' THEN metric_value END)::numeric, 4
            )                                                                                  AS tasa_clase1,
            MAX(CASE WHEN metric_name = 'class1_count' THEN metric_value END)::int            AS clase1_count,
            MAX(CASE WHEN metric_name = 'class0_count' THEN metric_value END)::int            AS clase0_count,
            ROUND(
                MAX(CASE WHEN metric_name = 'memory_mb'  THEN metric_value END)::numeric, 1
            )                                                                                  AS memoria_mb,
            MAX(CASE WHEN metric_name = 'null_count'  THEN metric_value END)::int             AS nulos_totales,
            MAX(CASE WHEN metric_name = 'inf_count'   THEN metric_value END)::int             AS inf_totales
        FROM debit_pipeline_metrics
        WHERE stage = 'feature_engineering'
        GROUP BY run_id, split_strategy, source_name
        ORDER BY source_name
    """)


def downgrade() -> None:
    for v in _VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {v} CASCADE")
    op.drop_table("debit_pipeline_metrics")
