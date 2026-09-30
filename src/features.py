"""Feature engineering and dataset assembly."""

import pandas as pd

from . import config
from . import data_loaders as dl


def exclude_covid(df: pd.DataFrame, start: str = config.COVID_START, end: str = config.COVID_END) -> pd.DataFrame:
    mask = (df.index >= start) & (df.index <= end)
    print(f"[exclude_covid] dropping {mask.sum()} hourly rows between {start} and {end}")
    return df.loc[~mask]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    return df


def build_dataset(rate_class: str, exclude_covid_window: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the full modeling dataset for one rate class.

    Returns (dataframe, feature_columns). Target column is 'load_mw'.
    """
    y = dl.load_target(rate_class)
    wx = dl.load_weather()
    date_cats = dl.load_date_categories()

    df = y.join(wx, how="left").join(date_cats, how="left")

    if exclude_covid_window:
        df = exclude_covid(df)

    df = add_time_features(df)

    feature_cols = ["temp_f", "dew_point_f", "humidity"] + list(date_cats.columns) + ["hour"]

    before = len(df)
    df = df.dropna(subset=feature_cols + ["load_mw"])
    dropped = before - len(df)
    if dropped:
        print(f"[build_dataset:{rate_class}] dropped {dropped} rows with missing features/target")

    return df, feature_cols
