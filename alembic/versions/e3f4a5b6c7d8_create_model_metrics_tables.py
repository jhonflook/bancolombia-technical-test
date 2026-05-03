"""create model metrics tables

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-05-02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "debit_model_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False, unique=True),
        sa.Column("parent_run_id", sa.String(64), nullable=True),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("split_strategy", sa.String(16), nullable=False),
        sa.Column("experiment_name", sa.String(128), nullable=False),
        sa.Column("run_date", sa.DateTime(timezone=True), nullable=False),
        # hiperparámetros y configuración
        sa.Column("n_features", sa.Integer, nullable=True),
        sa.Column("n_trials", sa.Integer, nullable=True),
        sa.Column("train_size", sa.Integer, nullable=True),
        sa.Column("test_size", sa.Integer, nullable=True),
        sa.Column("oot_size", sa.Integer, nullable=True),
        sa.Column("scale_pos_weight", sa.Float, nullable=True),
        sa.Column("best_params", JSONB, nullable=True),
        # métricas CV
        sa.Column("cv_auc", sa.Float, nullable=True),
        # métricas train
        sa.Column("train_auc", sa.Float, nullable=True),
        sa.Column("train_ks", sa.Float, nullable=True),
        sa.Column("train_f1", sa.Float, nullable=True),
        sa.Column("train_auc_pr", sa.Float, nullable=True),
        sa.Column("train_precision", sa.Float, nullable=True),
        sa.Column("train_recall", sa.Float, nullable=True),
        # métricas test
        sa.Column("test_auc", sa.Float, nullable=True),
        sa.Column("test_ks", sa.Float, nullable=True),
        sa.Column("test_f1", sa.Float, nullable=True),
        sa.Column("test_auc_pr", sa.Float, nullable=True),
        sa.Column("test_precision", sa.Float, nullable=True),
        sa.Column("test_recall", sa.Float, nullable=True),
        # métricas OOT
        sa.Column("oot_auc", sa.Float, nullable=True),
        sa.Column("oot_ks", sa.Float, nullable=True),
        sa.Column("oot_f1", sa.Float, nullable=True),
        sa.Column("oot_auc_pr", sa.Float, nullable=True),
        sa.Column("oot_precision", sa.Float, nullable=True),
        sa.Column("oot_recall", sa.Float, nullable=True),
        schema="public",
    )
    op.create_index("ix_model_metrics_model_name", "debit_model_metrics", ["model_name"])
    op.create_index("ix_model_metrics_split",      "debit_model_metrics", ["split_strategy"])
    op.create_index("ix_model_metrics_run_date",   "debit_model_metrics", ["run_date"])

    op.create_table(
        "debit_model_features",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("feature_name", sa.String(256), nullable=False),
        sa.Column("importance", sa.Float, nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        schema="public",
    )
    op.create_index("ix_model_features_run_id",    "debit_model_features", ["run_id"])
    op.create_index("ix_model_features_model_name","debit_model_features", ["model_name"])


def downgrade() -> None:
    op.drop_table("debit_model_features")
    op.drop_table("debit_model_metrics")
