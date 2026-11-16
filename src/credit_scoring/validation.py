"""Схема валидации."""

import pandas as pd


def time_split(
    df: pd.DataFrame, valid_size: float = 0.2, oot_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбивает заявки на train / valid / out-of-time по порядку id.

    Порядок id соответствует времени (см. EDA: adversarial ROC-AUC 0.80 между ранними и
    поздними заявками), поэтому валидация по времени ближе к реальному применению модели,
    чем случайное разбиение.

    args:
        df - датафрейм заявок, индекс - id.
        valid_size - доля заявок в valid (идёт сразу после train).
        oot_size - доля самых поздних заявок в out-of-time выборке.

    returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] - train, valid и oot части.
    """
    df = df.sort_index()
    n = len(df)
    valid_start = int(n * (1 - valid_size - oot_size))
    oot_start = int(n * (1 - oot_size))
    return df.iloc[:valid_start], df.iloc[valid_start:oot_start], df.iloc[oot_start:]
