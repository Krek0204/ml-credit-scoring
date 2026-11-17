"""Признаки крайних кредитов заявки по порядку rn.

По EDA направление rn по времени однозначно не установлено: корреляция rn с бинами
pre_since_opened почти нулевая, но у кредитов с большим rn чаще нет данных о платежах
24 месяца назад, то есть они, вероятно, новее. Поэтому функция умеет брать кредиты
с обоих концов, а выбор делается экспериментально.
"""

import pandas as pd

from credit_scoring.features.payments import loan_payment_summary, paym_matrix

RECENT_COLS = [
    "pre_since_opened", "pre_pterm", "pre_till_pclose",
    "pre_loans_credit_limit", "pre_loans_outstanding", "pre_loans_credit_cost_rate",
    "pre_util", "pre_over2limit", "pre_maxover2limit",
    "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060", "is_zero_loans6090", "is_zero_loans90",
    "enc_loans_account_holder_type", "enc_loans_credit_status", "enc_loans_credit_type",
]
# Для нескольких кредитов коды категорий не усредняем - у них нет порядка
RECENT_CAT_COLS = ["enc_loans_account_holder_type", "enc_loans_credit_status", "enc_loans_credit_type"]


def recent_loan_features(feature_df: pd.DataFrame, n_loans: int = 3, from_end: bool = True) -> pd.DataFrame:
    """Признаки одного крайнего кредита и средние по n_loans крайним кредитам заявки.

    args:
        feature_df - датафрейм с кредитами, индекс - id заявки.
        n_loans - сколько крайних кредитов усреднять.
        from_end - True: кредиты с наибольшим rn (префикс last), False: с наименьшим (префикс first).

    returns:
        pd.DataFrame - признаки заявки, индекс - id.
    """
    prefix = "last" if from_end else "first"

    loans = feature_df[["rn"] + RECENT_COLS].copy()
    paym = paym_matrix(feature_df)
    for name, values in loan_payment_summary(paym).items():
        loans[name] = values
    loans["paym_last1_status"] = paym[:, 0]
    del paym
    # Нужный крайний кредит должен оказаться первым внутри заявки
    loans = loans.reset_index().sort_values(["id", "rn"], ascending=[True, not from_end])

    # Крайний кредит - первая строка заявки после сортировки
    grouped = loans.groupby("id", sort=True)
    one = grouped.head(1).set_index("id").drop(columns="rn").add_prefix(f"{prefix}1_")

    num_cols = [c for c in loans.columns if c not in ("id", "rn", *RECENT_CAT_COLS)]
    several = grouped.head(n_loans).groupby("id")[num_cols].mean().add_prefix(f"{prefix}{n_loans}_")
    return pd.concat([one, several], axis=1)
