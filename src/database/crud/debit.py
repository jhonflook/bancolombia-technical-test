"""CRUD operations for debit recurrent analysis tables."""

import logging
from datetime import date

from sqlmodel import Session, func, select

from src.models.debit import DebitCanal, DebitClient, DebitExcedente, DebitGestion, DebitMora, DebitPago

logger = logging.getLogger(__name__)


def get_clients(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    var_rta: int | None = None,
) -> list[DebitClient]:
    """Retrieve debit clients optionally filtered by period range and target value.

    Parameters
    ----------
    session:
        Active SQLModel session.
    start_date:
        Lower bound for f_analisis (inclusive).
    end_date:
        Upper bound for f_analisis (inclusive).
    var_rta:
        Filter by target value (0 or 1). None returns both classes.

    Returns
    -------
    list[DebitClient]
        Matching client records ordered by f_analisis.
    """
    stmt = select(DebitClient)
    if start_date:
        stmt = stmt.where(DebitClient.f_analisis >= start_date)
    if end_date:
        stmt = stmt.where(DebitClient.f_analisis <= end_date)
    if var_rta is not None:
        stmt = stmt.where(DebitClient.var_rta == var_rta)
    stmt = stmt.order_by(DebitClient.f_analisis)
    return list(session.exec(stmt).all())


def get_target_distribution(session: Session) -> dict[int, int]:
    """Return count of records per target class across all periods.

    Parameters
    ----------
    session:
        Active SQLModel session.

    Returns
    -------
    dict[int, int]
        Mapping {var_rta_value: count}, e.g. {0: 9901, 1: 36835}.
    """
    stmt = select(DebitClient.var_rta, func.count(DebitClient.id)).group_by(DebitClient.var_rta)
    rows = session.exec(stmt).all()
    result = {int(var_rta): int(count) for var_rta, count in rows}
    logger.info("Target distribution: %s", result)
    return result


def get_periods(session: Session) -> list[date]:
    """Return sorted list of distinct f_analisis periods in the dataset.

    Parameters
    ----------
    session:
        Active SQLModel session.

    Returns
    -------
    list[date]
        Sorted list of available analysis periods.
    """
    stmt = select(DebitClient.f_analisis).distinct().order_by(DebitClient.f_analisis)
    return list(session.exec(stmt).all())


def get_pagos_by_period(
    session: Session,
    start_date: date,
    end_date: date,
) -> list[DebitPago]:
    """Retrieve payment records for a period range.

    Parameters
    ----------
    session:
        Active SQLModel session.
    start_date:
        Lower bound for f_analisis (inclusive).
    end_date:
        Upper bound for f_analisis (inclusive).

    Returns
    -------
    list[DebitPago]
        Payment records ordered by f_analisis.
    """
    stmt = (
        select(DebitPago)
        .where(DebitPago.f_analisis >= start_date)
        .where(DebitPago.f_analisis <= end_date)
        .order_by(DebitPago.f_analisis)
    )
    return list(session.exec(stmt).all())
