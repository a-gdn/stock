import pandas as pd

def pct(number: float) -> float:
    return round(number * 100, 2)

def get_rolling_min(close_df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return close_df.rolling(window=n_past_days, closed='left').min() # closed = 'left' excludes the last row (i.e. current row)

def get_rolling_max(close_df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return close_df.rolling(window=n_past_days, closed='left').max()

def get_pivot(rolling_max_df: pd.DataFrame, rolling_min_df: pd.DataFrame, close_df:pd.DataFrame) -> pd.DataFrame:
    return (rolling_max_df + rolling_min_df + close_df) / 3

def get_support1(pivot_df: pd.DataFrame, rolling_max_df: pd.DataFrame) -> pd.DataFrame:
    return (pivot_df * 2) - rolling_max_df

def get_support2(pivot_df: pd.DataFrame, rolling_max_df: pd.DataFrame, rolling_min_df: pd.DataFrame) -> pd.DataFrame:
    return pivot_df - (rolling_max_df - rolling_min_df)

def get_resistance1(pivot_df: pd.DataFrame, rolling_min_df: pd.DataFrame) -> pd.DataFrame:
    return (pivot_df * 2) - rolling_min_df

def get_resistance2(pivot_df: pd.DataFrame, rolling_max_df: pd.DataFrame, rolling_min_df: pd.DataFrame) -> pd.DataFrame:
    return pivot_df + (rolling_max_df - rolling_min_df)