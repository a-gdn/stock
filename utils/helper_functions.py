import pandas as pd

def _get_n_past_days(df: pd.DataFrame, column: str, n_past_days: int) -> pd.DataFrame:
    return df[::-1][column].rolling(n_past_days)

def get_rolling_min(df: pd.DataFrame, column: str, n_past_days: int) -> float:
    return _get_n_past_days(df, column, n_past_days).min()

def get_rolling_max(df: pd.DataFrame, column: str, n_past_days: int) -> float:
    return _get_n_past_days(df, column, n_past_days).max()

def get_pivot(rolling_max: float, rolling_min: float, close: float) -> float:
    return (rolling_max + rolling_min + close) / 3

def get_support1(pivot: float, rolling_max: float) -> float:
    return (pivot * 2) - rolling_max

def get_support2(pivot:float, rolling_max: float, rolling_min: float) -> float:
    return pivot - (rolling_max - rolling_min)

def get_resistance1(pivot: float, rolling_min: float) -> float:
    return (pivot * 2) - rolling_min

def get_resistance2(pivot: float, rolling_max: float, rolling_min: float) -> float:
    return pivot + (rolling_max - rolling_min)