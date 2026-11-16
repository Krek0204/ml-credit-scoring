"""Утилиты для чтения файлов."""

from pathlib import Path

import pandas as pd


def _downcast_ints(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит целочисленные колонки к минимальному подходящему типу (int8/int16/...).

    Все признаки датасета - закодированные значения в небольшом диапазоне, поэтому int64
    избыточен: даункаст уменьшает потребление памяти примерно в 8 раз.
    """
    int_cols = df.select_dtypes("integer").columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    return df


def read_features(data_path: str) -> pd.DataFrame:
    """Функция для чтения сырых данных датасета - всех parquet файлов из папки.

    args:
        data_path - строка, путь к папке с parquet файлами датасета.

    returns:
        pd.DataFrame - датафрейм с кредитами, индекс - id заявки.
    """
    # Сортируем по номеру файла, чтобы порядок чтения не зависел от файловой системы
    paths = sorted(Path(data_path).glob('*.pq'), key=lambda p: int(p.stem.split('_')[-1]))
    df_list = [_downcast_ints(pd.read_parquet(file_path, engine='fastparquet')) for file_path in paths]
    data_df = pd.concat(df_list, ignore_index=True).set_index("id")
    return data_df


def read_target(target_path: str) -> pd.DataFrame:
    """Функция для чтения csv файла с целевым признаком.

    args:
        target_path - строка, путь к файлу csv с целевым признаком.

    returns:
        pd.DataFrame - датафрейм с целевым признаком, индекс - id заявки.
    """
    return _downcast_ints(pd.read_csv(target_path, index_col='id'))


def read_dataset(data_path: str, target_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Функция для чтения датасета - сырых данных и целевого значения. Принимает путь к папке с parquet файлами и
    путь к csv файлу с целевым значением

    args:
        data_path - строка, путь к папке с parquet файлами датасета.
        target_path - строка, путь к файлу csv с целевым признаком.

    returns:
        tuple[pd.Dataframe, pd.DataFrame] - кортеж, состоящий из двух датафреймов pandas.
        Первый - это датафрейм с основными данными датасета. Второй - датафрейм с целевым признаком.
    """
    return (read_features(data_path), read_target(target_path))
