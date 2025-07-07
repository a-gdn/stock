import pandas as pd
import numpy as np
import pickle
import random
import string
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# GENERAL
def get_num_combinations(list_of_lists: list[list]) -> int:
    lengths = [len(sublist) for sublist in list_of_lists]
    num_combinations = 1
    
    for length in lengths:
        num_combinations *= length
    
    print(f'number of combinations: {num_combinations}')

    return num_combinations

def print_combination(current_combination: int, total_combinations: int):
    print(f'step: {current_combination}/{total_combinations}')

def get_num_combinations(param_dict: dict) -> None:
    num_combinations = 1
    for key, value in param_dict.items():
        num_combinations *= len(value)

    print('number of combinations:', num_combinations)

    return num_combinations 

def print_progress(current_iteration, total_iterations):
    if current_iteration <= 0:
        return "Invalid current iteration"

    if current_iteration > total_iterations:
        return "Current iteration cannot be greater than total iterations"

    percentage_completed = current_iteration / total_iterations

    current_time = datetime.datetime.now()
    estimated_remaining_time = (current_time - current_time * percentage_completed) / percentage_completed

    estimated_end_hour = current_time + estimated_remaining_time
    print(f"""\r
        step: {current_iteration}/{total_iterations}
        estimated_remaining_time: {estimated_remaining_time}
        estimated_end_hour: {estimated_end_hour}""", end='')

def get_date() -> str:
    return datetime.today().strftime('%Y-%m-%d')

def fillnavalues(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()

    df_cleaned.ffill(inplace=True) #forward fill otherwise
    # df_cleaned.bfill(inplace=True) #backward fill for the first rows
    df_cleaned.fillna(0, inplace=True) #fill remaining NaN values with 0
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(inplace=True) #drop rows with NaN values
    return df_cleaned

def save_object(obj, filename):
    """Save a Python object to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    """Load a Python object from a file using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
def get_random_string(length):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_string

# SUPPORT & RESISTANCE

def pct(number: float) -> float:
    return round(number * 100, 2)

def get_trimmed_average(df: pd.Series, pct_to_trim: float, min_num_to_trim: int) -> float:
    sorted_df = df.sort_values()
    num_to_trim = max(min_num_to_trim, round(len(sorted_df) * pct_to_trim))
    trimmed_df = sorted_df[:-num_to_trim]
    trimmed_average = trimmed_df.mean()
    return trimmed_average

def get_rolling_min(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return df.rolling(window=n_past_days, closed='left').min() # closed = 'left' excludes the last row (i.e. current row)

def get_rolling_max(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return df.rolling(window=n_past_days, closed='left').max()

def get_forward_rolling_df(df: pd.DataFrame, n_future_days: int, buying_time: str, selling_time: str) -> pd.DataFrame:
    if buying_time == 'Open' and selling_time == 'Open':
        window_size = n_future_days
    elif buying_time == 'Close' and selling_time == 'Open':
        df = df.shift(-1)
        window_size = n_future_days - 1
    elif buying_time == 'Open' and selling_time == 'Close':
        window_size = n_future_days + 1
    elif buying_time == 'Close' and selling_time == 'Close':
        df = df.shift(-1)
        window_size = n_future_days
    
    forward_window = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    return df.rolling(window=forward_window)
    
def get_future_rolling_min(min_df: pd.DataFrame, n_future_days: int, buying_time: str, selling_time: str) -> pd.DataFrame:
    forward_rolling_min_df = get_forward_rolling_df(min_df, n_future_days, buying_time, selling_time)
    return forward_rolling_min_df.min()

def get_future_rolling_max(max_df: pd.DataFrame, n_future_days: int, buying_time: str, selling_time: str) -> pd.DataFrame:
    forward_rolling_max_df = get_forward_rolling_df(max_df, n_future_days, buying_time, selling_time)
    return forward_rolling_max_df.max()

def get_future_rolling_max_position(max_df: pd.DataFrame, n_future_days: int, buying_time: str, selling_time: str) -> pd.DataFrame:
    forward_rolling_max_df = get_forward_rolling_df(max_df, n_future_days, buying_time, selling_time)
    rolling_max_position = forward_rolling_max_df.apply(lambda x: int(np.argmax(x)) if np.any(x) else np.nan, raw=True)
    return rolling_max_position

def get_future_rolling_min_value(row, col, min_df, n_future_days_df):
    n_future_days = n_future_days_df.iloc[row, col] + 1
    min_value = min_df.iloc[row:row + int(n_future_days), col].min() if not np.isnan(n_future_days) else np.nan
    return min_value

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


# REORGANIZE DATAFRAME
def remove_top_column_name(df):
    return df.droplevel(0, axis='columns')

def concat_dfs(df_list: list[pd.DataFrame], column_list: list[str]) -> pd.DataFrame:    
    df = pd.concat(df_list, axis=1, keys=column_list)
    return df.dropna()

def concat_dfs_different_indexes(df1: pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
    df2.index = df1.index
    return pd.concat([df1, df2], axis=1)

def get_last_characters_from_index(df: pd.DataFrame, n_last_characters: int) -> pd.DataFrame:
    return df.index.get_level_values(0).astype(str).str[:n_last_characters]

def filter_by_lower_and_upper_limits(df: pd.DataFrame, column: str, lower_limit: float, upper_limit: float) -> pd.DataFrame:
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

def get_rows_after_date(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    #start_date formatted as 'YYYY-MM-DD'
    return df[df.index >= pd.Timestamp(start_date)]

def rename_first_column(df, new_col_name: str) -> pd.DataFrame:
    df_renamed = df.copy()
    df_renamed = pd.DataFrame(df_renamed)
    df_renamed.rename(columns={df_renamed.columns[0] : new_col_name}, inplace=True)

    return df_renamed

def stack(df, new_col_name):
    df_stacked = df.stack(future_stack=True) # Stack columns, using future_stack is to be Pandas 3.0 compatible
    df_stacked = df_stacked.to_frame() # Convert to DataFrame
    df_stacked = rename_first_column(df_stacked, new_col_name) # Rename the 1st (and only) column

    return df_stacked

def expand(df_to_expand: pd.DataFrame, df_index: pd.DataFrame, index_name) -> pd.DataFrame:
    level_values = df_index.index.get_level_values(index_name)
    df_expanded = df_to_expand.reindex(level_values).set_index(df_index.index)
    
    df_expanded.ffill(inplace=True)
    df_expanded.bfill(inplace=True)
    
    return df_expanded

def create_floor_mask(df: pd.DataFrame, low_df:pd.DataFrame, loss_limit:float, n_future_days:int) -> pd.DataFrame:
    if loss_limit > 0:
        raise ValueError("loss_limit should be <= 0.")
    
    future_min_values = get_future_rolling_min(low_df, n_future_days)
    loss_threshold = df * (1 + (loss_limit / 100)) # loss_limit is neagtive
    
    return future_min_values < loss_threshold

def get_last_column(df):
    return df[df.columns[-1]]

def calculate_averages(df:pd.DataFrame) -> pd.DataFrame:
    return df.mean(axis='columns')

# DATAFRAME CALCULATION
def get_num_tickers(df: pd.DataFrame) -> int:
    return df.shape[1]

def ensure_dataframe(input_data):
    if isinstance(input_data, pd.Series):
        return input_data.to_frame()
    else:
        return input_data
    
def calculate_variations(df, n_past_days, n_future_days):
    var = df.shift(-n_future_days) / df.shift(n_past_days)
    var = ensure_dataframe(var)
    
    return var

def calculate_positive_rate(df: pd.DataFrame) -> float:
    positive_values_count = (df> 0).sum()
    total_count = len(df)

    return (positive_values_count / total_count) if total_count != 0 else float('nan')

def get_fee_coef(fee: float) -> pd.DataFrame:
    return (1 - fee) / (1 + fee)

def apply_fee(df: pd.DataFrame, fee: float) -> pd.DataFrame:
    return df * get_fee_coef(fee)

def calculate_volatility(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    pct_change = df.pct_change(fill_method=None)
    volatility = pct_change.rolling(window=n_past_days).std()
    return pd.DataFrame(volatility)

def calculate_n_ups(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    df_changes = df.diff()
    df_n_ups = (df_changes > 0).rolling(window=n_past_days).sum()
    return df_n_ups

def calculate_var_rank(df:pd.DataFrame, n_past_days:int, n_future_days:int) -> pd.DataFrame:
    df_var = calculate_variations(df, n_past_days, n_future_days)
    df_rank = calculate_rank(df_var)
    return df_rank

def calculate_rank(df):
    df_rank = df.rank(axis='columns', ascending=False)
    return df_rank

def calculate_performance_vs_market(df:pd.DataFrame, n_past_days:int) -> pd.DataFrame:
    df_var = calculate_variations(df, n_past_days, 0)
    row_average = calculate_averages(df_var)
    df_performance_vs_market = df.div(row_average, axis='index')
    return df_performance_vs_market

def calculate_market_variations(df:pd.DataFrame, n_past_days:int) -> pd.DataFrame:
    row_average = calculate_averages(df)
    row_average_var = calculate_variations(row_average, n_past_days, 0)
    return pd.DataFrame(row_average_var)

def get_days_since_min(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return n_past_days - df.rolling(window=n_past_days + 1).apply(lambda x: x.argmin(), raw=True)

def get_days_since_max(df: pd.DataFrame, n_past_days: int) -> pd.DataFrame:
    return n_past_days - df.rolling(window=n_past_days + 1).apply(lambda x: x.argmax(), raw=True)

def classify_var(df_var: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    def classify_with_max_values_first(value):
        for index, threshold in enumerate(sorted_threholds):
            if pd.isna(value):
                return pd.NA
            if value > threshold:
                return index
        return len(sorted_threholds)
    
    sorted_threholds = sorted(thresholds, reverse=True)
    df_class = df_var.map(classify_with_max_values_first)
    return df_class

def classify_rank(df_rank: pd.DataFrame, thresholds:list[float]) -> pd.DataFrame:
    def classify_with_min_values_first(value):
        for index, threshold in enumerate(sorted_threholds):
            if pd.isna(value):
                return pd.NA
            if value < threshold:
                return index
        return len(sorted_threholds)
    
    sorted_threholds = sorted(thresholds)
    df_class = df_rank.map(classify_with_min_values_first)

    return df_class

# INDICATORS

def validate_dataframe(df, function_name=""):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(f"Input DataFrame is empty in {function_name}.")

    if df.isnull().any().any():
        logging.warning(f"DataFrame contains NaN values in {function_name}. Consider handling them before calculation.")

def calculate_rsi(df, period=14):
    delta = df.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df.ewm(span=short_period, min_periods=1, adjust=False).mean()
    long_ema = df.ewm(span=long_period, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
    return macd, signal

def calculate_atr(df_high, df_low, df_close, period=14):
    high_low = df_high - df_low
    high_close = np.abs(df_high - df_close.shift(1))
    low_close = np.abs(df_low - df_close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_bollinger_bands(df, period=20):
    sma = df.rolling(window=period, min_periods=1).mean()
    std = df.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# SCIKIT LEARN
def print_report(y_real, y_prediction):
    print("Classification Report:")
    print(classification_report(y_real, y_prediction))

    conf_matrix = confusion_matrix(y_real, y_prediction)

    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]

    print(f"""
    True Positive (Acheté, bon choix): {tp}
    True Negative (Pas acheté, bon choix): {tn}
    False Positive (Acheté, mauvais choix): {fp}
    False Negative (Pas acheté, mauvais choix): {fn}
    """)

# FILES

def get_file_path(folder_path: str, file_name: str) -> str:
    date = get_date()
    return f'{folder_path}{file_name}_{date}'

def save_df_as_csv(df: pd.DataFrame, folder_path: str, file_name: str):
    file_path = get_file_path(folder_path, file_name)
    print('Saving file csv file...')
    df.to_csv(f'{file_path}.csv')

def save_df_as_xlsx(df: pd.DataFrame, folder_path: str, file_name: str):
    file_path = get_file_path(folder_path, file_name)
    print('Saving file xlsx file...')
    df.to_excel(f'{file_path}.xlsx')

# FINBOX

# Mapping between Yahoo suffixes and Finbox prefixes
YAHOO_SUFFIX_TO_FINBOX_PREFIX = {
    ".BR": "ENXTBR:",
    ".MC": "BME:",
    ".MI": "BIT:",
    ".OL": "OB:",
    ".AS": "ENXTAM:",
    ".PA": "ENXTPA:"
}

# Reverse mapping from Finbox prefixes to Yahoo suffixes
FINBOX_PREFIX_TO_YAHOO_SUFFIX = {
    finbox_prefix: yahoo_suffix 
    for yahoo_suffix, finbox_prefix in YAHOO_SUFFIX_TO_FINBOX_PREFIX.items()
}

def yahoo_to_finbox(yahoo_ticker):
    for yahoo_suffix, finbox_prefix in YAHOO_SUFFIX_TO_FINBOX_PREFIX.items():
        if yahoo_ticker.endswith(yahoo_suffix):
            base_ticker = yahoo_ticker.removesuffix(yahoo_suffix)
            return f"{finbox_prefix}{base_ticker}"
    raise ValueError(f"Unrecognized Yahoo ticker format: {yahoo_ticker}")

def finbox_to_yahoo(finbox_ticker):
    for finbox_prefix, yahoo_suffix in FINBOX_PREFIX_TO_YAHOO_SUFFIX.items():
        if finbox_ticker.startswith(finbox_prefix):
            base_ticker = finbox_ticker.removeprefix(finbox_prefix)
            return f"{base_ticker}{yahoo_suffix}"
    raise ValueError(f"Unrecognized Finbox ticker format: {finbox_ticker}")

