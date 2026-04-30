"""SQLModel models for debit recurrent analysis (prueba técnica Bancolombia).

Primary key: (num_doc, obl17, f_analisis) — client–obligation–period.
canales features stored as JSONB because the source has 4,258 columns,
which exceeds PostgreSQL's 1,600 column limit per table.
"""

from datetime import date
from typing import Any, Optional

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class DebitClient(SQLModel, table=True):
    """Master table with response variable.

    var_rta=1: obligation paid exclusively by debit with recurrence >= 40%.
    var_rta=0: payments through other channels or channel combinations.
    """

    __tablename__ = "debit_clients"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_clients_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    var_rta: int


class DebitExcedente(SQLModel, table=True):
    """Payment surplus and percentage paid over quota per time window."""

    __tablename__ = "debit_excedentes"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_excedentes_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    # excedente_pago (nulls grow with window — young obligations without history)
    avg_excedente_pago_3m: Optional[float] = None
    min_excedente_pago_3m: Optional[float] = None
    max_excedente_pago_3m: Optional[float] = None
    stddev_excedente_pago_3m: Optional[float] = None
    avg_excedente_pago_6m: Optional[float] = None
    min_excedente_pago_6m: Optional[float] = None
    max_excedente_pago_6m: Optional[float] = None
    stddev_excedente_pago_6m: Optional[float] = None
    avg_excedente_pago_9m: Optional[float] = None
    min_excedente_pago_9m: Optional[float] = None
    max_excedente_pago_9m: Optional[float] = None
    stddev_excedente_pago_9m: Optional[float] = None
    avg_excedente_pago_12m: Optional[float] = None
    min_excedente_pago_12m: Optional[float] = None
    max_excedente_pago_12m: Optional[float] = None
    stddev_excedente_pago_12m: Optional[float] = None
    # porc_pago
    avg_porc_pago_3m: Optional[float] = None
    min_porc_pago_3m: Optional[float] = None
    max_porc_pago_3m: Optional[float] = None
    stddev_porc_pago_3m: Optional[float] = None
    avg_porc_pago_6m: Optional[float] = None
    min_porc_pago_6m: Optional[float] = None
    max_porc_pago_6m: Optional[float] = None
    stddev_porc_pago_6m: Optional[float] = None
    avg_porc_pago_9m: Optional[float] = None
    min_porc_pago_9m: Optional[float] = None
    max_porc_pago_9m: Optional[float] = None
    stddev_porc_pago_9m: Optional[float] = None
    avg_porc_pago_12m: Optional[float] = None
    min_porc_pago_12m: Optional[float] = None
    max_porc_pago_12m: Optional[float] = None
    stddev_porc_pago_12m: Optional[float] = None


class DebitGestion(SQLModel, table=True):
    """Collection management actions: agreements, RPC, promises, rank."""

    __tablename__ = "debit_gestiones"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_gestiones_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    # cant_gestiones
    avg_cant_gestiones_3m: Optional[float] = None
    min_cant_gestiones_3m: Optional[float] = None
    max_cant_gestiones_3m: Optional[float] = None
    stddev_cant_gestiones_3m: Optional[float] = None
    avg_cant_gestiones_6m: Optional[float] = None
    min_cant_gestiones_6m: Optional[float] = None
    max_cant_gestiones_6m: Optional[float] = None
    stddev_cant_gestiones_6m: Optional[float] = None
    avg_cant_gestiones_9m: Optional[float] = None
    min_cant_gestiones_9m: Optional[float] = None
    max_cant_gestiones_9m: Optional[float] = None
    stddev_cant_gestiones_9m: Optional[float] = None
    avg_cant_gestiones_12m: Optional[float] = None
    min_cant_gestiones_12m: Optional[float] = None
    max_cant_gestiones_12m: Optional[float] = None
    stddev_cant_gestiones_12m: Optional[float] = None
    # cant_rpc
    avg_cant_rpc_3m: Optional[float] = None
    min_cant_rpc_3m: Optional[float] = None
    max_cant_rpc_3m: Optional[float] = None
    stddev_cant_rpc_3m: Optional[float] = None
    avg_cant_rpc_6m: Optional[float] = None
    min_cant_rpc_6m: Optional[float] = None
    max_cant_rpc_6m: Optional[float] = None
    stddev_cant_rpc_6m: Optional[float] = None
    avg_cant_rpc_9m: Optional[float] = None
    min_cant_rpc_9m: Optional[float] = None
    max_cant_rpc_9m: Optional[float] = None
    stddev_cant_rpc_9m: Optional[float] = None
    avg_cant_rpc_12m: Optional[float] = None
    min_cant_rpc_12m: Optional[float] = None
    max_cant_rpc_12m: Optional[float] = None
    stddev_cant_rpc_12m: Optional[float] = None
    # cant_acuerdo
    avg_cant_acuerdo_3m: Optional[float] = None
    min_cant_acuerdo_3m: Optional[float] = None
    max_cant_acuerdo_3m: Optional[float] = None
    stddev_cant_acuerdo_3m: Optional[float] = None
    avg_cant_acuerdo_6m: Optional[float] = None
    min_cant_acuerdo_6m: Optional[float] = None
    max_cant_acuerdo_6m: Optional[float] = None
    stddev_cant_acuerdo_6m: Optional[float] = None
    avg_cant_acuerdo_9m: Optional[float] = None
    min_cant_acuerdo_9m: Optional[float] = None
    max_cant_acuerdo_9m: Optional[float] = None
    stddev_cant_acuerdo_9m: Optional[float] = None
    avg_cant_acuerdo_12m: Optional[float] = None
    min_cant_acuerdo_12m: Optional[float] = None
    max_cant_acuerdo_12m: Optional[float] = None
    stddev_cant_acuerdo_12m: Optional[float] = None
    # promesas_cumplidas
    avg_promesas_cumplidas_3m: Optional[float] = None
    min_promesas_cumplidas_3m: Optional[float] = None
    max_promesas_cumplidas_3m: Optional[float] = None
    stddev_promesas_cumplidas_3m: Optional[float] = None
    avg_promesas_cumplidas_6m: Optional[float] = None
    min_promesas_cumplidas_6m: Optional[float] = None
    max_promesas_cumplidas_6m: Optional[float] = None
    stddev_promesas_cumplidas_6m: Optional[float] = None
    avg_promesas_cumplidas_9m: Optional[float] = None
    min_promesas_cumplidas_9m: Optional[float] = None
    max_promesas_cumplidas_9m: Optional[float] = None
    stddev_promesas_cumplidas_9m: Optional[float] = None
    avg_promesas_cumplidas_12m: Optional[float] = None
    min_promesas_cumplidas_12m: Optional[float] = None
    max_promesas_cumplidas_12m: Optional[float] = None
    stddev_promesas_cumplidas_12m: Optional[float] = None
    # maximo_rank
    avg_maximo_rank_3m: Optional[float] = None
    min_maximo_rank_3m: Optional[float] = None
    max_maximo_rank_3m: Optional[float] = None
    stddev_maximo_rank_3m: Optional[float] = None
    avg_maximo_rank_6m: Optional[float] = None
    min_maximo_rank_6m: Optional[float] = None
    max_maximo_rank_6m: Optional[float] = None
    stddev_maximo_rank_6m: Optional[float] = None
    avg_maximo_rank_9m: Optional[float] = None
    min_maximo_rank_9m: Optional[float] = None
    max_maximo_rank_9m: Optional[float] = None
    stddev_maximo_rank_9m: Optional[float] = None
    avg_maximo_rank_12m: Optional[float] = None
    min_maximo_rank_12m: Optional[float] = None
    max_maximo_rank_12m: Optional[float] = None
    stddev_maximo_rank_12m: Optional[float] = None


class DebitMora(SQLModel, table=True):
    """Delinquency days statistics per time window.

    Note: 9m window only has avg (no min/max/stddev in source data).
    12m window absent from source — treated as missing feature (S6).
    """

    __tablename__ = "debit_moras"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_moras_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    moras_avg_mora_3m: Optional[float] = None
    moras_min_mora_3m: Optional[float] = None
    moras_max_mora_3m: Optional[float] = None
    moras_stddev_mora_3m: Optional[float] = None
    moras_avg_mora_6m: Optional[float] = None
    moras_min_mora_6m: Optional[float] = None
    moras_max_mora_6m: Optional[float] = None
    moras_stddev_mora_6m: Optional[float] = None
    moras_avg_mora_9m: Optional[float] = None  # only avg available for 9m


class DebitPago(SQLModel, table=True):
    """Payment aggregates by channel type (debit, physical, virtual, other).

    LEAKAGE RISK: pago_debito is a direct signal of the target variable.
    Ensure these features are computed from periods prior to the labeling period.
    """

    __tablename__ = "debit_pagos"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_pagos_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    # debito
    avg_pago_debito_3m: Optional[float] = None
    min_pago_debito_3m: Optional[float] = None
    max_pago_debito_3m: Optional[float] = None
    stddev_pago_debito_3m: Optional[float] = None
    avg_pago_debito_6m: Optional[float] = None
    min_pago_debito_6m: Optional[float] = None
    max_pago_debito_6m: Optional[float] = None
    stddev_pago_debito_6m: Optional[float] = None
    avg_pago_debito_9m: Optional[float] = None
    min_pago_debito_9m: Optional[float] = None
    max_pago_debito_9m: Optional[float] = None
    stddev_pago_debito_9m: Optional[float] = None
    avg_pago_debito_12m: Optional[float] = None
    min_pago_debito_12m: Optional[float] = None
    max_pago_debito_12m: Optional[float] = None
    stddev_pago_debito_12m: Optional[float] = None
    # fisico
    avg_pago_fisico_3m: Optional[float] = None
    min_pago_fisico_3m: Optional[float] = None
    max_pago_fisico_3m: Optional[float] = None
    stddev_pago_fisico_3m: Optional[float] = None
    avg_pago_fisico_6m: Optional[float] = None
    min_pago_fisico_6m: Optional[float] = None
    max_pago_fisico_6m: Optional[float] = None
    stddev_pago_fisico_6m: Optional[float] = None
    avg_pago_fisico_9m: Optional[float] = None
    min_pago_fisico_9m: Optional[float] = None
    max_pago_fisico_9m: Optional[float] = None
    stddev_pago_fisico_9m: Optional[float] = None
    avg_pago_fisico_12m: Optional[float] = None
    min_pago_fisico_12m: Optional[float] = None
    max_pago_fisico_12m: Optional[float] = None
    stddev_pago_fisico_12m: Optional[float] = None
    # virtual
    avg_pago_virtual_3m: Optional[float] = None
    min_pago_virtual_3m: Optional[float] = None
    max_pago_virtual_3m: Optional[float] = None
    stddev_pago_virtual_3m: Optional[float] = None
    avg_pago_virtual_6m: Optional[float] = None
    min_pago_virtual_6m: Optional[float] = None
    max_pago_virtual_6m: Optional[float] = None
    stddev_pago_virtual_6m: Optional[float] = None
    avg_pago_virtual_9m: Optional[float] = None
    min_pago_virtual_9m: Optional[float] = None
    max_pago_virtual_9m: Optional[float] = None
    stddev_pago_virtual_9m: Optional[float] = None
    avg_pago_virtual_12m: Optional[float] = None
    min_pago_virtual_12m: Optional[float] = None
    max_pago_virtual_12m: Optional[float] = None
    stddev_pago_virtual_12m: Optional[float] = None
    # otros
    avg_pago_otros_3m: Optional[float] = None
    min_pago_otros_3m: Optional[float] = None
    max_pago_otros_3m: Optional[float] = None
    stddev_pago_otros_3m: Optional[float] = None
    avg_pago_otros_6m: Optional[float] = None
    min_pago_otros_6m: Optional[float] = None
    max_pago_otros_6m: Optional[float] = None
    stddev_pago_otros_6m: Optional[float] = None
    avg_pago_otros_9m: Optional[float] = None
    min_pago_otros_9m: Optional[float] = None
    max_pago_otros_9m: Optional[float] = None
    stddev_pago_otros_9m: Optional[float] = None
    avg_pago_otros_12m: Optional[float] = None
    min_pago_otros_12m: Optional[float] = None
    max_pago_otros_12m: Optional[float] = None
    stddev_pago_otros_12m: Optional[float] = None


class DebitCanal(SQLModel, table=True):
    """Transactional data by channel (app, cajero, SVE, etc.).

    Source has 4,258 columns — exceeds PostgreSQL's 1,600 column limit.
    Summary totals stored as typed columns; all channel-detail features
    stored in JSONB for flexibility and zero-loss fidelity (S4: absent
    channels = no transactional activity → impute 0 at read time).
    """

    __tablename__ = "debit_canales"
    __table_args__ = (
        UniqueConstraint("num_doc", "obl17", "f_analisis", name="uq_debit_canales_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    num_doc: str = Field(index=True)
    obl17: str = Field(index=True)
    f_analisis: date = Field(index=True)
    trx_mnt_total: Optional[float] = None
    trx_mnt_total_smmlv: Optional[float] = None
    trx_cnt_total: Optional[float] = None
    # All remaining ~4,252 channel×type features stored as JSONB
    features: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
