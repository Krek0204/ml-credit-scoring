import pandas as pd

# Числовые признаки - mean / max / sum / std
NUM_COLS = [
    "pre_since_opened", "pre_since_confirmed", "pre_pterm", "pre_fterm",
    "pre_till_pclose", "pre_till_fclose",
    "pre_loans_credit_limit", "pre_loans_next_pay_summ", "pre_loans_outstanding",
    "pre_loans_total_overdue", "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
    "pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90",
    "pre_util", "pre_over2limit", "pre_maxover2limit",
]

# Бинарные флаги - доля 1 по заявке
BIN_COLS = [
    "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060",
    "is_zero_loans6090", "is_zero_loans90",
    "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit",
    "pclose_flag", "fclose_flag",
]

# Категории - mode (самый частый) и nunique
CAT_COLS = [
    "enc_loans_account_holder_type", "enc_loans_credit_status",
    "enc_loans_credit_type", "enc_loans_account_cur",
]


def _mode_by_id(feature_df: pd.DataFrame, col: str) -> pd.Series:
    """Самое частое значение признака для каждой заявки.

    Векторный аналог groupby(...).agg(lambda s: s.mode().iloc[0]): при равенстве частот
    берётся наименьшее значение, как и в Series.mode().
    """
    counts = feature_df.groupby(["id", col]).size().rename("cnt").reset_index()
    counts = counts.sort_values(["id", "cnt", col], ascending=[True, False, True])
    return counts.drop_duplicates("id").set_index("id")[col]


def build_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует кредиты до уровня заявки: одна строка - один id.

    args:
        feature_df - датафрейм с кредитами, id в индексе или в колонке.

    returns:
        pd.DataFrame - датафрейм с признаками заявки, индекс - id. Таргет не используется,
        поэтому функция одинаково применима к train и test.
    """
    if "id" not in feature_df.columns:
        feature_df = feature_df.reset_index()
    grouped = feature_df.groupby("id")

    agg = grouped.agg(n_loans=("rn", "count"))

    num_agg = grouped[NUM_COLS].agg(["mean", "max", "sum", "std"])
    num_agg.columns = [f"{c}_{stat}" for c, stat in num_agg.columns]

    bin_agg = grouped[BIN_COLS].mean()
    bin_agg.columns = [f"{c}_mean" for c in bin_agg.columns]

    cat_parts = {}
    for c in CAT_COLS:
        cat_parts[f"{c}_nunique"] = grouped[c].nunique()
        cat_parts[f"{c}_mode_value"] = _mode_by_id(feature_df, c)
    cat_agg = pd.DataFrame(cat_parts)

    # Платежные статусы - mean/max по месяцам
    paym_cols = [c for c in feature_df.columns if c.startswith("enc_paym_")]
    paym_agg = grouped[paym_cols].agg(["mean", "max"])
    paym_agg.columns = [f"{c}_{stat}" for c, stat in paym_agg.columns]

    features = pd.concat([agg, num_agg, bin_agg, cat_agg, paym_agg], axis=1)
    features.index.name = "id"
    return features


def aggregate(feature_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует кредиты и присоединяет таргет - удобная обёртка для тренировочной выборки."""
    return target_df.join(build_features(feature_df), how="left")
