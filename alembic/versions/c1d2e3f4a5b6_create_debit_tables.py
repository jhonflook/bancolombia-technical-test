"""create_debit_tables

Revision ID: c1d2e3f4a5b6
Revises:
Create Date: 2025-04-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the 6 debit_* tables for the recurrent debit analysis project."""
    op.create_table(
        "debit_clients",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("var_rta", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_clients_key"),
    )
    op.create_index("ix_debit_clients_num_doc", "debit_clients", ["num_doc"])
    op.create_index("ix_debit_clients_obl17", "debit_clients", ["obl17"])
    op.create_index("ix_debit_clients_f_analisis", "debit_clients", ["f_analisis"])

    op.create_table(
        "debit_excedentes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("avg_excedente_pago_3m", sa.Float(), nullable=True),
        sa.Column("min_excedente_pago_3m", sa.Float(), nullable=True),
        sa.Column("max_excedente_pago_3m", sa.Float(), nullable=True),
        sa.Column("stddev_excedente_pago_3m", sa.Float(), nullable=True),
        sa.Column("avg_excedente_pago_6m", sa.Float(), nullable=True),
        sa.Column("min_excedente_pago_6m", sa.Float(), nullable=True),
        sa.Column("max_excedente_pago_6m", sa.Float(), nullable=True),
        sa.Column("stddev_excedente_pago_6m", sa.Float(), nullable=True),
        sa.Column("avg_excedente_pago_9m", sa.Float(), nullable=True),
        sa.Column("min_excedente_pago_9m", sa.Float(), nullable=True),
        sa.Column("max_excedente_pago_9m", sa.Float(), nullable=True),
        sa.Column("stddev_excedente_pago_9m", sa.Float(), nullable=True),
        sa.Column("avg_excedente_pago_12m", sa.Float(), nullable=True),
        sa.Column("min_excedente_pago_12m", sa.Float(), nullable=True),
        sa.Column("max_excedente_pago_12m", sa.Float(), nullable=True),
        sa.Column("stddev_excedente_pago_12m", sa.Float(), nullable=True),
        sa.Column("avg_porc_pago_3m", sa.Float(), nullable=True),
        sa.Column("min_porc_pago_3m", sa.Float(), nullable=True),
        sa.Column("max_porc_pago_3m", sa.Float(), nullable=True),
        sa.Column("stddev_porc_pago_3m", sa.Float(), nullable=True),
        sa.Column("avg_porc_pago_6m", sa.Float(), nullable=True),
        sa.Column("min_porc_pago_6m", sa.Float(), nullable=True),
        sa.Column("max_porc_pago_6m", sa.Float(), nullable=True),
        sa.Column("stddev_porc_pago_6m", sa.Float(), nullable=True),
        sa.Column("avg_porc_pago_9m", sa.Float(), nullable=True),
        sa.Column("min_porc_pago_9m", sa.Float(), nullable=True),
        sa.Column("max_porc_pago_9m", sa.Float(), nullable=True),
        sa.Column("stddev_porc_pago_9m", sa.Float(), nullable=True),
        sa.Column("avg_porc_pago_12m", sa.Float(), nullable=True),
        sa.Column("min_porc_pago_12m", sa.Float(), nullable=True),
        sa.Column("max_porc_pago_12m", sa.Float(), nullable=True),
        sa.Column("stddev_porc_pago_12m", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_excedentes_key"),
    )
    op.create_index("ix_debit_excedentes_num_doc", "debit_excedentes", ["num_doc"])
    op.create_index("ix_debit_excedentes_obl17", "debit_excedentes", ["obl17"])
    op.create_index("ix_debit_excedentes_f_analisis", "debit_excedentes", ["f_analisis"])

    op.create_table(
        "debit_gestiones",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("avg_cant_gestiones_3m", sa.Float(), nullable=True),
        sa.Column("min_cant_gestiones_3m", sa.Float(), nullable=True),
        sa.Column("max_cant_gestiones_3m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_gestiones_3m", sa.Float(), nullable=True),
        sa.Column("avg_cant_gestiones_6m", sa.Float(), nullable=True),
        sa.Column("min_cant_gestiones_6m", sa.Float(), nullable=True),
        sa.Column("max_cant_gestiones_6m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_gestiones_6m", sa.Float(), nullable=True),
        sa.Column("avg_cant_gestiones_9m", sa.Float(), nullable=True),
        sa.Column("min_cant_gestiones_9m", sa.Float(), nullable=True),
        sa.Column("max_cant_gestiones_9m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_gestiones_9m", sa.Float(), nullable=True),
        sa.Column("avg_cant_gestiones_12m", sa.Float(), nullable=True),
        sa.Column("min_cant_gestiones_12m", sa.Float(), nullable=True),
        sa.Column("max_cant_gestiones_12m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_gestiones_12m", sa.Float(), nullable=True),
        sa.Column("avg_cant_rpc_3m", sa.Float(), nullable=True),
        sa.Column("min_cant_rpc_3m", sa.Float(), nullable=True),
        sa.Column("max_cant_rpc_3m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_rpc_3m", sa.Float(), nullable=True),
        sa.Column("avg_cant_rpc_6m", sa.Float(), nullable=True),
        sa.Column("min_cant_rpc_6m", sa.Float(), nullable=True),
        sa.Column("max_cant_rpc_6m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_rpc_6m", sa.Float(), nullable=True),
        sa.Column("avg_cant_rpc_9m", sa.Float(), nullable=True),
        sa.Column("min_cant_rpc_9m", sa.Float(), nullable=True),
        sa.Column("max_cant_rpc_9m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_rpc_9m", sa.Float(), nullable=True),
        sa.Column("avg_cant_rpc_12m", sa.Float(), nullable=True),
        sa.Column("min_cant_rpc_12m", sa.Float(), nullable=True),
        sa.Column("max_cant_rpc_12m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_rpc_12m", sa.Float(), nullable=True),
        sa.Column("avg_cant_acuerdo_3m", sa.Float(), nullable=True),
        sa.Column("min_cant_acuerdo_3m", sa.Float(), nullable=True),
        sa.Column("max_cant_acuerdo_3m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_acuerdo_3m", sa.Float(), nullable=True),
        sa.Column("avg_cant_acuerdo_6m", sa.Float(), nullable=True),
        sa.Column("min_cant_acuerdo_6m", sa.Float(), nullable=True),
        sa.Column("max_cant_acuerdo_6m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_acuerdo_6m", sa.Float(), nullable=True),
        sa.Column("avg_cant_acuerdo_9m", sa.Float(), nullable=True),
        sa.Column("min_cant_acuerdo_9m", sa.Float(), nullable=True),
        sa.Column("max_cant_acuerdo_9m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_acuerdo_9m", sa.Float(), nullable=True),
        sa.Column("avg_cant_acuerdo_12m", sa.Float(), nullable=True),
        sa.Column("min_cant_acuerdo_12m", sa.Float(), nullable=True),
        sa.Column("max_cant_acuerdo_12m", sa.Float(), nullable=True),
        sa.Column("stddev_cant_acuerdo_12m", sa.Float(), nullable=True),
        sa.Column("avg_promesas_cumplidas_3m", sa.Float(), nullable=True),
        sa.Column("min_promesas_cumplidas_3m", sa.Float(), nullable=True),
        sa.Column("max_promesas_cumplidas_3m", sa.Float(), nullable=True),
        sa.Column("stddev_promesas_cumplidas_3m", sa.Float(), nullable=True),
        sa.Column("avg_promesas_cumplidas_6m", sa.Float(), nullable=True),
        sa.Column("min_promesas_cumplidas_6m", sa.Float(), nullable=True),
        sa.Column("max_promesas_cumplidas_6m", sa.Float(), nullable=True),
        sa.Column("stddev_promesas_cumplidas_6m", sa.Float(), nullable=True),
        sa.Column("avg_promesas_cumplidas_9m", sa.Float(), nullable=True),
        sa.Column("min_promesas_cumplidas_9m", sa.Float(), nullable=True),
        sa.Column("max_promesas_cumplidas_9m", sa.Float(), nullable=True),
        sa.Column("stddev_promesas_cumplidas_9m", sa.Float(), nullable=True),
        sa.Column("avg_promesas_cumplidas_12m", sa.Float(), nullable=True),
        sa.Column("min_promesas_cumplidas_12m", sa.Float(), nullable=True),
        sa.Column("max_promesas_cumplidas_12m", sa.Float(), nullable=True),
        sa.Column("stddev_promesas_cumplidas_12m", sa.Float(), nullable=True),
        sa.Column("avg_maximo_rank_3m", sa.Float(), nullable=True),
        sa.Column("min_maximo_rank_3m", sa.Float(), nullable=True),
        sa.Column("max_maximo_rank_3m", sa.Float(), nullable=True),
        sa.Column("stddev_maximo_rank_3m", sa.Float(), nullable=True),
        sa.Column("avg_maximo_rank_6m", sa.Float(), nullable=True),
        sa.Column("min_maximo_rank_6m", sa.Float(), nullable=True),
        sa.Column("max_maximo_rank_6m", sa.Float(), nullable=True),
        sa.Column("stddev_maximo_rank_6m", sa.Float(), nullable=True),
        sa.Column("avg_maximo_rank_9m", sa.Float(), nullable=True),
        sa.Column("min_maximo_rank_9m", sa.Float(), nullable=True),
        sa.Column("max_maximo_rank_9m", sa.Float(), nullable=True),
        sa.Column("stddev_maximo_rank_9m", sa.Float(), nullable=True),
        sa.Column("avg_maximo_rank_12m", sa.Float(), nullable=True),
        sa.Column("min_maximo_rank_12m", sa.Float(), nullable=True),
        sa.Column("max_maximo_rank_12m", sa.Float(), nullable=True),
        sa.Column("stddev_maximo_rank_12m", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_gestiones_key"),
    )
    op.create_index("ix_debit_gestiones_num_doc", "debit_gestiones", ["num_doc"])
    op.create_index("ix_debit_gestiones_obl17", "debit_gestiones", ["obl17"])
    op.create_index("ix_debit_gestiones_f_analisis", "debit_gestiones", ["f_analisis"])

    op.create_table(
        "debit_moras",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("moras_avg_mora_3m", sa.Float(), nullable=True),
        sa.Column("moras_min_mora_3m", sa.Float(), nullable=True),
        sa.Column("moras_max_mora_3m", sa.Float(), nullable=True),
        sa.Column("moras_stddev_mora_3m", sa.Float(), nullable=True),
        sa.Column("moras_avg_mora_6m", sa.Float(), nullable=True),
        sa.Column("moras_min_mora_6m", sa.Float(), nullable=True),
        sa.Column("moras_max_mora_6m", sa.Float(), nullable=True),
        sa.Column("moras_stddev_mora_6m", sa.Float(), nullable=True),
        sa.Column("moras_avg_mora_9m", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_moras_key"),
    )
    op.create_index("ix_debit_moras_num_doc", "debit_moras", ["num_doc"])
    op.create_index("ix_debit_moras_obl17", "debit_moras", ["obl17"])
    op.create_index("ix_debit_moras_f_analisis", "debit_moras", ["f_analisis"])

    op.create_table(
        "debit_pagos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("avg_pago_debito_3m", sa.Float(), nullable=True),
        sa.Column("min_pago_debito_3m", sa.Float(), nullable=True),
        sa.Column("max_pago_debito_3m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_debito_3m", sa.Float(), nullable=True),
        sa.Column("avg_pago_debito_6m", sa.Float(), nullable=True),
        sa.Column("min_pago_debito_6m", sa.Float(), nullable=True),
        sa.Column("max_pago_debito_6m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_debito_6m", sa.Float(), nullable=True),
        sa.Column("avg_pago_debito_9m", sa.Float(), nullable=True),
        sa.Column("min_pago_debito_9m", sa.Float(), nullable=True),
        sa.Column("max_pago_debito_9m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_debito_9m", sa.Float(), nullable=True),
        sa.Column("avg_pago_debito_12m", sa.Float(), nullable=True),
        sa.Column("min_pago_debito_12m", sa.Float(), nullable=True),
        sa.Column("max_pago_debito_12m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_debito_12m", sa.Float(), nullable=True),
        sa.Column("avg_pago_fisico_3m", sa.Float(), nullable=True),
        sa.Column("min_pago_fisico_3m", sa.Float(), nullable=True),
        sa.Column("max_pago_fisico_3m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_fisico_3m", sa.Float(), nullable=True),
        sa.Column("avg_pago_fisico_6m", sa.Float(), nullable=True),
        sa.Column("min_pago_fisico_6m", sa.Float(), nullable=True),
        sa.Column("max_pago_fisico_6m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_fisico_6m", sa.Float(), nullable=True),
        sa.Column("avg_pago_fisico_9m", sa.Float(), nullable=True),
        sa.Column("min_pago_fisico_9m", sa.Float(), nullable=True),
        sa.Column("max_pago_fisico_9m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_fisico_9m", sa.Float(), nullable=True),
        sa.Column("avg_pago_fisico_12m", sa.Float(), nullable=True),
        sa.Column("min_pago_fisico_12m", sa.Float(), nullable=True),
        sa.Column("max_pago_fisico_12m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_fisico_12m", sa.Float(), nullable=True),
        sa.Column("avg_pago_virtual_3m", sa.Float(), nullable=True),
        sa.Column("min_pago_virtual_3m", sa.Float(), nullable=True),
        sa.Column("max_pago_virtual_3m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_virtual_3m", sa.Float(), nullable=True),
        sa.Column("avg_pago_virtual_6m", sa.Float(), nullable=True),
        sa.Column("min_pago_virtual_6m", sa.Float(), nullable=True),
        sa.Column("max_pago_virtual_6m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_virtual_6m", sa.Float(), nullable=True),
        sa.Column("avg_pago_virtual_9m", sa.Float(), nullable=True),
        sa.Column("min_pago_virtual_9m", sa.Float(), nullable=True),
        sa.Column("max_pago_virtual_9m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_virtual_9m", sa.Float(), nullable=True),
        sa.Column("avg_pago_virtual_12m", sa.Float(), nullable=True),
        sa.Column("min_pago_virtual_12m", sa.Float(), nullable=True),
        sa.Column("max_pago_virtual_12m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_virtual_12m", sa.Float(), nullable=True),
        sa.Column("avg_pago_otros_3m", sa.Float(), nullable=True),
        sa.Column("min_pago_otros_3m", sa.Float(), nullable=True),
        sa.Column("max_pago_otros_3m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_otros_3m", sa.Float(), nullable=True),
        sa.Column("avg_pago_otros_6m", sa.Float(), nullable=True),
        sa.Column("min_pago_otros_6m", sa.Float(), nullable=True),
        sa.Column("max_pago_otros_6m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_otros_6m", sa.Float(), nullable=True),
        sa.Column("avg_pago_otros_9m", sa.Float(), nullable=True),
        sa.Column("min_pago_otros_9m", sa.Float(), nullable=True),
        sa.Column("max_pago_otros_9m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_otros_9m", sa.Float(), nullable=True),
        sa.Column("avg_pago_otros_12m", sa.Float(), nullable=True),
        sa.Column("min_pago_otros_12m", sa.Float(), nullable=True),
        sa.Column("max_pago_otros_12m", sa.Float(), nullable=True),
        sa.Column("stddev_pago_otros_12m", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_pagos_key"),
    )
    op.create_index("ix_debit_pagos_num_doc", "debit_pagos", ["num_doc"])
    op.create_index("ix_debit_pagos_obl17", "debit_pagos", ["obl17"])
    op.create_index("ix_debit_pagos_f_analisis", "debit_pagos", ["f_analisis"])

    op.create_table(
        "debit_canales",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("num_doc", sa.String(), nullable=False),
        sa.Column("obl17", sa.String(), nullable=False),
        sa.Column("f_analisis", sa.Date(), nullable=False),
        sa.Column("trx_mnt_total", sa.Float(), nullable=True),
        sa.Column("trx_mnt_total_smmlv", sa.Float(), nullable=True),
        sa.Column("trx_cnt_total", sa.Float(), nullable=True),
        sa.Column("features", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_canales_key"),
    )
    op.create_index("ix_debit_canales_num_doc", "debit_canales", ["num_doc"])
    op.create_index("ix_debit_canales_obl17", "debit_canales", ["obl17"])
    op.create_index("ix_debit_canales_f_analisis", "debit_canales", ["f_analisis"])


def downgrade() -> None:
    """Drop all debit_* tables."""
    op.drop_table("debit_canales")
    op.drop_table("debit_pagos")
    op.drop_table("debit_moras")
    op.drop_table("debit_gestiones")
    op.drop_table("debit_excedentes")
    op.drop_table("debit_clients")
