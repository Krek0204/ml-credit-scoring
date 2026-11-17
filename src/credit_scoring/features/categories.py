"""Признаки-доли значений категориальных признаков и бинов по заявке."""

import numpy as np
import pandas as pd

# Значения зафиксированы по train и test, чтобы у обеих выборок были одинаковые колонки
CATEGORY_VALUES = {
    "enc_loans_account_holder_type": list(range(7)),
    "enc_loans_credit_status": list(range(7)),
    "enc_loans_credit_type": list(range(8)),
    "enc_loans_account_cur": list(range(4)),
}

# Бины pre_ признаков с немонотонной связью с таргетом (см. EDA) из числа самых важных
BIN_VALUES = {
    "pre_util": list(range(20)),
    "pre_loans_credit_cost_rate": list(range(14)),
    "pre_loans_outstanding": list(range(1, 6)),
    "pre_over2limit": list(range(20)),
    "pre_maxover2limit": list(range(20)),
}


def _value_shares(feature_df: pd.DataFrame, col: str, values: list[int], with_counts: bool) -> pd.DataFrame:
    """Доля (и при необходимости число) кредитов заявки с каждым значением признака."""
    column = feature_df[col].to_numpy()
    loan = pd.DataFrame(
        {f"{col}_{v}": (column == v).astype(np.float32) for v in values},
        index=feature_df.index,
    )
    grouped = loan.groupby(level="id")
    parts = [grouped.mean().add_suffix("_share")]
    if with_counts:
        parts.append(grouped.sum().add_suffix("_count"))
    return pd.concat(parts, axis=1)


def category_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Доли и количества кредитов каждого значения категориальных признаков по заявке.

    В отличие от mode, сохраняет информацию о редких, но рискованных значениях:
    например, один кредит со статусом 5 из десяти.
    """
    return pd.concat(
        [_value_shares(feature_df, col, values, with_counts=True) for col, values in CATEGORY_VALUES.items()],
        axis=1,
    )


def bin_share_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Доли кредитов заявки в каждом бине для выбранных pre_ признаков.

    Номера бинов не всегда упорядочены по риску, поэтому mean по номеру бина размывает
    рискованные бины - доли сохраняют их.
    """
    return pd.concat(
        [_value_shares(feature_df, col, values, with_counts=False) for col, values in BIN_VALUES.items()],
        axis=1,
    )
