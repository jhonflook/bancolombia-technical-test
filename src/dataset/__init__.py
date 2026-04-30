"""Dataset module — modelo analítico de débitos recurrentes."""

from src.dataset.data_preparation import (
    JOIN_KEYS,
    OOT_START,
    TARGET_COL,
    TEST_END,
    TEST_START,
    TRAIN_END,
    build_analytical_model,
    get_feature_columns,
    load_raw_sources,
    split_train_test_oot,
    validate_join_integrity,
)

__all__ = [
    # Constantes
    "JOIN_KEYS",
    "TARGET_COL",
    "TRAIN_END",
    "TEST_START",
    "TEST_END",
    "OOT_START",
    # Pipeline principal
    "load_raw_sources",
    "build_analytical_model",
    "validate_join_integrity",
    "split_train_test_oot",
    "get_feature_columns",
]
