"""Признаки по статусам ежемесячных платежей enc_paym_0..24.

Месяц 0 - самый свежий. По EDA статусы закодированы так: 0 - платёж без просрочки,
1 и 2 - просрочка (2 - самый рискованный), 3 - нет данных (в 97.6% кредитов статус 3
идёт сплошным хвостом в старых месяцах, то есть кредит ещё не был открыт).
"""

import numpy as np
import pandas as pd

PAYM_COLS = [f"enc_paym_{i}" for i in range(25)]
# В этих месяцах статусы закодированы со сдвигом на +1 (значения 1-4 вместо 0-3)
SHIFTED_PAYM_COLS = ["enc_paym_11", "enc_paym_20", "enc_paym_24"]

STATUSES = [0, 1, 2, 3]
OVERDUE_STATUSES = [1, 2]
NO_DATA_STATUS = 3

# Окна по давности: количество последних месяцев
WINDOWS = {"last1": 1, "last3": 3, "last6": 6, "last12": 12, "all": 25}


def paym_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    """Матрица статусов платежей (кредиты x 25 месяцев) с исправленной кодировкой."""
    paym = feature_df[PAYM_COLS].to_numpy(dtype=np.int8, copy=True)
    for col in SHIFTED_PAYM_COLS:
        paym[:, PAYM_COLS.index(col)] -= 1
    return paym


def loan_payment_summary(paym: np.ndarray) -> dict[str, np.ndarray]:
    """Сводка по платежам на уровне кредита - используется и здесь, и в признаках последних кредитов."""
    is_overdue = np.isin(paym, OVERDUE_STATUSES)
    has_overdue = is_overdue.any(axis=1)
    return {
        "paym_months_with_data": (paym != NO_DATA_STATUS).sum(axis=1).astype(np.float32),
        "paym_overdue_last12": is_overdue[:, :12].sum(axis=1).astype(np.float32),
        # Месяцев с последней (самой свежей) просрочки; NaN, если просрочек не было
        "paym_months_since_overdue": np.where(has_overdue, is_overdue.argmax(axis=1), np.nan).astype(np.float32),
    }


def payment_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Доли статусов платежей в окнах по давности и признаки просрочек, агрегированные по заявке.

    args:
        feature_df - датафрейм с кредитами, индекс - id заявки.

    returns:
        pd.DataFrame - признаки заявки, индекс - id.
    """
    ids = feature_df.index
    paym = paym_matrix(feature_df)
    parts = []

    # Доли статусов и доля просрочек среди месяцев с данными - по каждому окну отдельно,
    # чтобы не держать в памяти все кредитные колонки сразу
    for name, n_months in WINDOWS.items():
        window = paym[:, :n_months]
        loan = {f"paym_{name}_s{s}_share": (window == s).mean(axis=1, dtype=np.float32) for s in STATUSES}
        with_data = (window != NO_DATA_STATUS).sum(axis=1)
        overdue = np.isin(window, OVERDUE_STATUSES).sum(axis=1)
        loan[f"paym_{name}_overdue_rate"] = np.where(
            with_data > 0, overdue / np.maximum(with_data, 1), np.nan
        ).astype(np.float32)
        agg = pd.DataFrame(loan, index=ids).groupby(level="id").agg(["mean", "max"])
        agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
        parts.append(agg)

    summary = pd.DataFrame(loan_payment_summary(paym), index=ids).groupby(level="id").agg({
        "paym_months_with_data": ["mean", "max"],
        "paym_overdue_last12": ["sum", "max"],
        # Самая свежая просрочка среди всех кредитов заявки
        "paym_months_since_overdue": ["min"],
    })
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    parts.append(summary)

    return pd.concat(parts, axis=1)
