"""CRUD operations module."""

from .debit import (
    get_clients,
    get_pagos_by_period,
    get_periods,
    get_target_distribution,
)

__all__ = [
    "get_clients",
    "get_target_distribution",
    "get_periods",
    "get_pagos_by_period",
]
