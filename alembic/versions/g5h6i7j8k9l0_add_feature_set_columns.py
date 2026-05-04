"""add feature_set columns to model metrics tables

Revision ID: g5h6i7j8k9l0
Revises: f4g5h6i7j8k9
Create Date: 2026-05-03
"""

from alembic import op
import sqlalchemy as sa

revision = "g5h6i7j8k9l0"
down_revision = "f4g5h6i7j8k9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # debit_model_metrics: agregar feature_set
    op.add_column(
        "debit_model_metrics",
        sa.Column("feature_set", sa.String(8), nullable=False, server_default="A"),
    )
    op.create_index(
        "ix_model_metrics_feature_set", "debit_model_metrics", ["feature_set"]
    )

    # debit_model_features: agregar split_strategy + feature_set para filtrado
    # directo en Metabase sin JOIN a debit_model_metrics
    op.add_column(
        "debit_model_features",
        sa.Column("split_strategy", sa.String(16), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "debit_model_features",
        sa.Column("feature_set", sa.String(8), nullable=False, server_default="A"),
    )
    op.create_index(
        "ix_model_features_feature_set", "debit_model_features", ["feature_set"]
    )


def downgrade() -> None:
    op.drop_index("ix_model_features_feature_set", table_name="debit_model_features")
    op.drop_column("debit_model_features", "feature_set")
    op.drop_column("debit_model_features", "split_strategy")

    op.drop_index("ix_model_metrics_feature_set", table_name="debit_model_metrics")
    op.drop_column("debit_model_metrics", "feature_set")
