import utils.helper_functions as hf
import config as cfg

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def calculate_rsi(df, period=14):
    hf.validate_dataframe(df, function_name="calculate_rsi")

    rsi = hf.calculate_rsi(df, period)
    return hf.stack(rsi, f'input_rsi_{period}d')

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    hf.validate_dataframe(df, function_name="calculate_macd")

    macd, signal = hf.calculate_macd(df, short_period, long_period, signal_period)
    return hf.stack(macd, 'input_macd'), hf.stack(signal, 'input_macd_signal')

def calculate_atr(df_stock_high, df_stock_low, df_stock_close, period=14):
    hf.validate_dataframe(df_stock_high, function_name="calculate_atr - df_stock_high")
    hf.validate_dataframe(df_stock_low, function_name="calculate_atr - df_stock_low")
    hf.validate_dataframe(df_stock_close, function_name="calculate_atr - df_stock_close")

    atr = hf.calculate_atr(df_stock_high, df_stock_low, df_stock_close, period)
    atr = hf.ensure_dataframe(atr)
    return hf.stack(atr, f'input_atr_{period}d')

def calculate_bollinger_bands(df, period=20):
    upper_band, lower_band = hf.calculate_bollinger_bands(df, period)
    upper_band_stacked = hf.stack(upper_band, f'input_bollinger_upper_{period}d')
    lower_band_stacked = hf.stack(lower_band, f'input_bollinger_lower_{period}d')
    return upper_band_stacked, lower_band_stacked

def get_market_wide_indicators(df_sp500, df_vix, df_correlation):
    sp500_stacked = hf.stack(df_sp500, 'input_sp500')
    vix_stacked = hf.stack(df_vix, 'input_vix')
    correlation_stacked = hf.stack(df_correlation, 'input_market_correlation')
    return sp500_stacked, vix_stacked, correlation_stacked


def calculate_stock_var(df, past_days, col_name):
    var = hf.calculate_variations(df, past_days, n_future_days=0)
    var_stacked = hf.stack(var, col_name)

    return var_stacked

def calculate_market_var(df, df_index, past_days, col_name):
    var = hf.calculate_variations(df, past_days, n_future_days=0)
    var = hf.rename_first_column(var, new_col_name=col_name)
    var_expanded = hf.expand(var, df_index, index_name='Date')

    return var_expanded

def calculate_stock_var_vs_past_ohlcv(df, df_past, past_days, col_name):
    var = df / df_past.shift(past_days)
    var_stacked = hf.stack(var, col_name)

    return var_stacked

def calculate_market_var_vs_past_ohlcv(df_stock, df_market_past, past_days, col_name):
    var = df_stock.div(df_market_past.shift(past_days), axis='index')
    var_stacked = hf.stack(var, col_name)

    return var_stacked

def clean_volume_data(df_stock_volume):
    df_cleaned_volume = df_stock_volume.copy()
    df_cleaned_volume.replace(0, 1e-6, inplace=True) 
    df_cleaned_volume.replace([np.inf, -np.inf], [1e6, -1e6], inplace=True)

    return df_cleaned_volume

def get_volume(df_stock_volume, past_day):
    df_cleaned_volume = clean_volume_data(df_stock_volume)
    volume_var = df_cleaned_volume.shift(past_day)

    volume_var_stacked = hf.stack(volume_var, f'input_volume_{past_day}d')

    return volume_var_stacked

def calculate_volume_var(df_stock_volume, past_start_day, past_end_day):
    df_cleaned_volume = clean_volume_data(df_stock_volume)

    volume_var = df_cleaned_volume.shift(past_end_day) / df_cleaned_volume.shift(past_start_day)
    volume_var_stacked = hf.stack(volume_var, f'input_volume_var_{past_start_day}-{past_end_day}d')

    return volume_var_stacked

def min_max_var(df, past_days):
    rolling_min = df.rolling(window=past_days + 1, min_periods=1).min()
    min_var = df / rolling_min
    min_var_stacked = hf.stack(min_var, f'input_min_var_past_{past_days}d')

    rolling_max = df.rolling(window=past_days + 1, min_periods=1).max()
    max_var = df / rolling_max
    max_var_stacked = hf.stack(max_var, f'input_max_var_past_{past_days}d')

    return min_var_stacked, max_var_stacked

def days_since_min_max(df, past_days):
    days_since_min = hf.get_days_since_min(df, past_days)
    days_since_min_stacked = hf.stack(days_since_min, f'input_days_since_min_{past_days}d')

    days_since_max = hf.get_days_since_max(df, past_days)
    days_since_max_stacked = hf.stack(days_since_max, f'input_days_since_max_{past_days}d')

    return days_since_min_stacked, days_since_max_stacked

def get_volatility(df, past_days):
    volatility = hf.calculate_volatility(df, past_days)
    volatility_stacked = hf.stack(volatility, f'input_volatility_{past_days}d')

    return volatility_stacked

def get_market_volatility(df, past_days):
    market_average = hf.calculate_averages(df)
    volatility = hf.calculate_volatility(market_average, past_days)
    volatility_stacked = hf.stack(volatility, f'input_market_volatility_{past_days}d')

    return volatility_stacked

def get_volume_volability(df, past_start_day, past_end_day):
    df_shifted = df.shift(past_end_day)
    window_size = past_start_day - past_end_day + 1
    volatility = hf.calculate_volatility(df_shifted, window_size)
    volatility_stacked = hf.stack(volatility, f'input_volume_volatility_{past_start_day}-{past_end_day}d')

    return volatility_stacked

def get_stock_n_ups(df, past_days):
    n_ups = hf.calculate_n_ups(df, past_days)
    n_ups_stacked = hf.stack(n_ups, f'input_stock_n_ups_{past_days}d')

    return n_ups_stacked

def get_market_n_ups(df, df_index, past_days, col_name):
    n_ups = hf.calculate_n_ups(df, past_days)
    n_ups = hf.rename_first_column(n_ups, new_col_name=col_name)
    n_ups_expanded = hf.expand(n_ups, df_index, index_name='Date')

    return n_ups_expanded

def get_performance_vs_market(df, past_days):
    performance_vs_market = hf.calculate_performance_vs_market(df, past_days)
    performance_vs_market_stacked = hf.stack(performance_vs_market, f'input_perf_vs_market_{past_days}d')

    return performance_vs_market_stacked

def get_var_rank(df, past_days):
    rank = hf.calculate_var_rank(df, past_days, n_future_days=0)
        
    return hf.stack(rank, f'input_rank_{past_days}d')

def get_rank(df, col_name):
    df_rank = hf.calculate_rank(df)
    df_rank = hf.stack(df_rank, col_name)

    return df_rank

def format_ref(df, col_name, df_index):
    df = hf.ensure_dataframe(df)
    df = hf.rename_first_column(df, new_col_name=col_name)
    df_reindexed = df_index.join(df, on='Date', how='left') # Apply index from another df

    return df_reindexed

def calculate_ref_var(df, past_days, col_name, df_index):
    var = hf.calculate_variations(df, past_days, n_future_days=0)
    var_reindexed = format_ref(var, col_name, df_index)

    return var_reindexed

def get_market_name(df):
    market_name = df.copy()
    for col in market_name.columns:
        market_name[col] = col[-2:]
    
    market_name_stacked = hf.stack(df, 'input_market_name')
    le = LabelEncoder()
    market_name_encoded = market_name_stacked.apply(le.fit_transform)

    return market_name_encoded

def get_inputs(dfs, buying_time):
    var_stock_90 = calculate_stock_var(dfs['df_stock_buy'], past_days=90, col_name='input_stock_var_90d')
    var_stock_30 = calculate_stock_var(dfs['df_stock_buy'], past_days=30, col_name='input_stock_var_30d')
    var_stock_10 = calculate_stock_var(dfs['df_stock_buy'], past_days=10, col_name='input_stock_var_10d')
    var_stock_1 = calculate_stock_var(dfs['df_stock_buy'], past_days=1, col_name='input_stock_var_1d')

    # var_brussels_market_90 = calculate_market_var(dfs['df_market_buy']['^BFX'], var_stock_90, past_days=90, col_name='input_brussels_market_var_90d')
    # var_brussels_market_30 = calculate_market_var(dfs['df_market_buy']['^BFX'], var_stock_90, past_days=30, col_name='input_brussels_market_var_30d')
    # var_brussels_market_10 = calculate_market_var(dfs['df_market_buy']['^BFX'], var_stock_90, past_days=10, col_name='input_brussels_market_var_10d')
    # var_brussels_market_1 = calculate_market_var(dfs['df_market_buy']['^BFX'], var_stock_90, past_days=1, col_name='input_brussels_market_var_1d')
    # var_madrid_market_90 = calculate_market_var(dfs['df_market_buy']['^IBEX'], var_stock_90, past_days=90, col_name='input_madrid_market_var_90d')
    # var_madrid_market_30 = calculate_market_var(dfs['df_market_buy']['^IBEX'], var_stock_90, past_days=30, col_name='input_madrid_market_var_30d')
    # var_madrid_market_10 = calculate_market_var(dfs['df_market_buy']['^IBEX'], var_stock_90, past_days=10, col_name='input_madrid_market_var_10d')
    # var_madrid_market_1 = calculate_market_var(dfs['df_market_buy']['^IBEX'], var_stock_90, past_days=1, col_name='input_madrid_market_var_1d')
    # var_milan_market_90 = calculate_market_var(dfs['df_market_buy']['FTSEMIB.MI'], var_stock_90, past_days=90, col_name='input_milan_market_var_90d')
    # var_milan_market_30 = calculate_market_var(dfs['df_market_buy']['FTSEMIB.MI'], var_stock_90, past_days=30, col_name='input_milan_market_var_30d')
    # var_milan_market_10 = calculate_market_var(dfs['df_market_buy']['FTSEMIB.MI'], var_stock_90, past_days=10, col_name='input_milan_market_var_10d')
    # var_milan_market_1 = calculate_market_var(dfs['df_market_buy']['FTSEMIB.MI'], var_stock_90, past_days=1, col_name='input_milan_market_var_1d')
    # var_nordic_market_90 = calculate_market_var(dfs['df_market_buy']['^OMX'], var_stock_90, past_days=90, col_name='input_nordic_market_var_90d')
    # var_nordic_market_30 = calculate_market_var(dfs['df_market_buy']['^OMX'], var_stock_90, past_days=30, col_name='input_nordic_market_var_30d')
    # var_nordic_market_10 = calculate_market_var(dfs['df_market_buy']['^OMX'], var_stock_90, past_days=10, col_name='input_nordic_market_var_10d')
    # var_nordic_market_1 = calculate_market_var(dfs['df_market_buy']['^OMX'], var_stock_90, past_days=1, col_name='input_nordic_market_var_1d')
    # var_amsterdam_market_90 = calculate_market_var(dfs['df_market_buy']['^AEX'], var_stock_90, past_days=90, col_name='input_amsterdam_market_var_90d')
    # var_amsterdam_market_30 = calculate_market_var(dfs['df_market_buy']['^AEX'], var_stock_90, past_days=30, col_name='input_amsterdam_market_var_30d')
    # var_amsterdam_market_10 = calculate_market_var(dfs['df_market_buy']['^AEX'], var_stock_90, past_days=10, col_name='input_amsterdam_market_var_10d')
    # var_amsterdam_market_1 = calculate_market_var(dfs['df_market_buy']['^AEX'], var_stock_90, past_days=1, col_name='input_amsterdam_market_var_1d')
    # var_paris_market_90 = calculate_market_var(dfs['df_market_buy']['^FCHI'], var_stock_90, past_days=90, col_name='input_paris_market_var_90d')
    # var_paris_market_30 = calculate_market_var(dfs['df_market_buy']['^FCHI'], var_stock_90, past_days=30, col_name='input_paris_market_var_30d')
    # var_paris_market_10 = calculate_market_var(dfs['df_market_buy']['^FCHI'], var_stock_90, past_days=10, col_name='input_paris_market_var_10d')
    # var_paris_market_1 = calculate_market_var(dfs['df_market_buy']['^FCHI'], var_stock_90, past_days=1, col_name='input_paris_market_var_1d')
    var_sp500_market_90 = calculate_market_var(dfs['df_market_buy']['^GSPC'], var_stock_90, past_days=90, col_name='input_sp500_market_var_90d')
    var_sp500_market_30 = calculate_market_var(dfs['df_market_buy']['^GSPC'], var_stock_90, past_days=30, col_name='input_sp500_market_var_30d')
    var_sp500_market_10 = calculate_market_var(dfs['df_market_buy']['^GSPC'], var_stock_90, past_days=10, col_name='input_sp500_market_var_10d')
    var_sp500_market_1 = calculate_market_var(dfs['df_market_buy']['^GSPC'], var_stock_90, past_days=1, col_name='input_sp500_market_var_1d')
    var_euro_market_90 = calculate_market_var(dfs['df_market_buy']['^STOXX50E'], var_stock_90, past_days=90, col_name='input_euro_market_var_90d')
    var_euro_market_30 = calculate_market_var(dfs['df_market_buy']['^STOXX50E'], var_stock_90, past_days=30, col_name='input_euro_market_var_30d')
    var_euro_market_10 = calculate_market_var(dfs['df_market_buy']['^STOXX50E'], var_stock_90, past_days=10, col_name='input_euro_market_var_10d')
    var_euro_market_1 = calculate_market_var(dfs['df_market_buy']['^STOXX50E'], var_stock_90, past_days=1, col_name='input_euro_market_var_1d')
    # var_frankfurt_market_90 = calculate_market_var(dfs['df_market_buy']['^GDAXI'], var_stock_90, past_days=90, col_name='input_frankfurt_market_var_90d')
    # var_frankfurt_market_30 = calculate_market_var(dfs['df_market_buy']['^GDAXI'], var_stock_90, past_days=30, col_name='input_frankfurt_market_var_30d')
    # var_frankfurt_market_10 = calculate_market_var(dfs['df_market_buy']['^GDAXI'], var_stock_90, past_days=10, col_name='input_frankfurt_market_var_10d')
    # var_frankfurt_market_1 = calculate_market_var(dfs['df_market_buy']['^GDAXI'], var_stock_90, past_days=1, col_name='input_frankfurt_market_var_1d')
    # var_london_market_90 = calculate_market_var(dfs['df_market_buy']['^FTSE'], var_stock_90, past_days=90, col_name='input_london_market_var_90d')
    # var_london_market_30 = calculate_market_var(dfs['df_market_buy']['^FTSE'], var_stock_90, past_days=30, col_name='input_london_market_var_30d')
    # var_london_market_10 = calculate_market_var(dfs['df_market_buy']['^FTSE'], var_stock_90, past_days=10, col_name='input_london_market_var_10d')
    # var_london_market_1 = calculate_market_var(dfs['df_market_buy']['^FTSE'], var_stock_90, past_days=1, col_name='input_london_market_var_1d')
    var_vix_90 = calculate_market_var(dfs['df_market_buy']['^VIX'], var_stock_90, past_days=90, col_name='input_vix_var_90d')
    var_vix_30 = calculate_market_var(dfs['df_market_buy']['^VIX'], var_stock_90, past_days=30, col_name='input_vix_var_30d')
    var_vix_10 = calculate_market_var(dfs['df_market_buy']['^VIX'], var_stock_90, past_days=10, col_name='input_vix_var_10d')
    var_vix_1 = calculate_market_var(dfs['df_market_buy']['^VIX'], var_stock_90, past_days=1, col_name='input_vix_var_1d')

    # market_name = get_market_name(dfs['df_stock_buy'])

    var_vs_stock_close_1 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_close'], past_days=1, col_name='input_stock_var_vs_close_1d')
    var_vs_stock_low_1 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_low'], past_days=1, col_name='input_stock_var_vs_low_1d')
    var_vs_stock_high_1 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_high'], past_days=1, col_name='input_stock_var_vs_high_1d')
    
    # var_vs_brussels_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^BFX'], past_days=1, col_name='input_brussels_var_vs_close_1d')
    # var_vs_brussels_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^BFX'], past_days=1, col_name='input_brussels_var_vs_low_1d')
    # var_vs_brussels_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^BFX'], past_days=1, col_name='input_brussels_var_vs_high_1d')
    # var_vs_madrid_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^IBEX'], past_days=1, col_name='input_madrid_var_vs_close_1d')
    # var_vs_madrid_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^IBEX'], past_days=1, col_name='input_madrid_var_vs_low_1d')
    # var_vs_madrid_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^IBEX'], past_days=1, col_name='input_madrid_var_vs_high_1d')
    # var_vs_milan_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['FTSEMIB.MI'], past_days=1, col_name='input_milan_var_vs_close_1d')
    # var_vs_milan_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['FTSEMIB.MI'], past_days=1, col_name='input_milan_var_vs_low_1d')
    # var_vs_milan_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['FTSEMIB.MI'], past_days=1, col_name='input_milan_var_vs_high_1d')
    # var_vs_nordic_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^OMX'], past_days=1, col_name='input_nordic_var_vs_close_1d')
    # var_vs_nordic_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^OMX'], past_days=1, col_name='input_nordic_var_vs_low_1d')
    # var_vs_nordic_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^OMX'], past_days=1, col_name='input_nordic_var_vs_high_1d')
    # var_vs_amsterdam_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^AEX'], past_days=1, col_name='input_amsterdam_var_vs_close_1d')
    # var_vs_amsterdam_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^AEX'], past_days=1, col_name='input_amsterdam_var_vs_low_1d')
    # var_vs_amsterdam_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^AEX'], past_days=1, col_name='input_amsterdam_var_vs_high_1d')
    # var_vs_paris_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^FCHI'], past_days=1, col_name='input_paris_var_vs_close_1d')
    # var_vs_paris_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^FCHI'], past_days=1, col_name='input_paris_var_vs_low_1d')
    # var_vs_paris_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^FCHI'], past_days=1, col_name='input_paris_var_vs_high_1d')
    var_vs_sp500_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^GSPC'], past_days=1, col_name='input_sp500_var_vs_close_1d')
    var_vs_sp500_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^GSPC'], past_days=1, col_name='input_sp500_var_vs_low_1d')
    var_vs_sp500_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^GSPC'], past_days=1, col_name='input_sp500_var_vs_high_1d')
    var_vs_euro_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^STOXX50E'], past_days=1, col_name='input_euro_var_vs_close_1d')
    var_vs_euro_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^STOXX50E'], past_days=1, col_name='input_euro_var_vs_low_1d')
    var_vs_euro_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^STOXX50E'], past_days=1, col_name='input_euro_var_vs_high_1d')
    # var_vs_frankfurt_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^GDAXI'], past_days=1, col_name='input_frankfurt_var_vs_close_1d')
    # var_vs_frankfurt_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^GDAXI'], past_days=1, col_name='input_frankfurt_var_vs_low_1d')
    # var_vs_frankfurt_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^GDAXI'], past_days=1, col_name='input_frankfurt_var_vs_high_1d')
    # var_vs_london_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^FTSE'], past_days=1, col_name='input_london_var_vs_close_1d')
    # var_vs_london_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^FTSE'], past_days=1, col_name='input_london_var_vs_low_1d')
    # var_vs_london_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^FTSE'], past_days=1, col_name='input_london_var_vs_high_1d')
    var_vs_vix_close_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_close']['^VIX'], past_days=1, col_name='input_vix_var_vs_close_1d')
    var_vs_vix_low_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_low']['^VIX'], past_days=1, col_name='input_vix_var_vs_low_1d')
    var_vs_vix_high_1 = calculate_market_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_market_high']['^VIX'], past_days=1, col_name='input_vix_var_vs_high_1d')

    # volume_1 = get_volume(dfs['df_stock_volume'], past_day=1)
    
    volume_var_90_1 = calculate_volume_var(dfs['df_stock_volume'], past_start_day=90, past_end_day=1)
    volume_var_30_1 = calculate_volume_var(dfs['df_stock_volume'], past_start_day=30, past_end_day=1)
    volume_var_10_1 = calculate_volume_var(dfs['df_stock_volume'], past_start_day=10, past_end_day=1)
    volume_var_2_1 = calculate_volume_var(dfs['df_stock_volume'], past_start_day=2, past_end_day=1)
    
    min_var_90, max_var_90 = min_max_var(dfs['df_stock_buy'], past_days=90)
    min_var_30, max_var_30 = min_max_var(dfs['df_stock_buy'], past_days=30)
    min_var_10, max_var_10 = min_max_var(dfs['df_stock_buy'], past_days=10)

    days_since_min_30, days_since_max_30 = days_since_min_max(dfs['df_stock_buy'], past_days=30)
    days_since_min_10, days_since_max_10 = days_since_min_max(dfs['df_stock_buy'], past_days=10)

    volatility_30 = get_volatility(dfs['df_stock_buy'], past_days=30)
    volatility_10 = get_volatility(dfs['df_stock_buy'], past_days=10)
    volatility_2 = get_volatility(dfs['df_stock_buy'], past_days=2)

    # market_volatility_30 = get_market_volatility(dfs['df_stock_buy'], past_days=30)
    # market_volatility_10 = get_market_volatility(dfs['df_stock_buy'], past_days=10)
    # market_volatility_2 = get_market_volatility(dfs['df_stock_buy'], past_days=2)

    volume_volability_90_1 = get_volume_volability(dfs['df_stock_volume'], past_start_day=90, past_end_day=1)
    volume_volability_30_1 = get_volume_volability(dfs['df_stock_volume'], past_start_day=30, past_end_day=1)
    volume_volability_10_1 = get_volume_volability(dfs['df_stock_volume'], past_start_day=10, past_end_day=1)
    volume_volability_2_1 = get_volume_volability(dfs['df_stock_volume'], past_start_day=2, past_end_day=1)

    stock_n_ups_90 = get_stock_n_ups(dfs['df_stock_buy'], past_days=90)
    stock_n_ups_30 = get_stock_n_ups(dfs['df_stock_buy'], past_days=30)
    stock_n_ups_5 = get_stock_n_ups(dfs['df_stock_buy'], past_days=5)

    # brussels_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^BFX'], stock_n_ups_90, past_days=90, col_name='input_brussels_n_ups_90d')
    # brussels_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^BFX'], stock_n_ups_30, past_days=30, col_name='input_brussels_n_ups_30d')
    # brussels_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^BFX'], stock_n_ups_5, past_days=5, col_name='input_brussels_n_ups_5d')
    # madrid_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^IBEX'], stock_n_ups_90, past_days=90, col_name='input_madrid_n_ups_90d')
    # madrid_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^IBEX'], stock_n_ups_30, past_days=30, col_name='input_madrid_n_ups_30d')
    # madrid_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^IBEX'], stock_n_ups_5, past_days=5, col_name='input_madrid_n_ups_5d')
    # milan_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['FTSEMIB.MI'], stock_n_ups_90, past_days=90, col_name='input_milan_n_ups_90d')
    # milan_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['FTSEMIB.MI'], stock_n_ups_30, past_days=30, col_name='input_milan_n_ups_30d')
    # milan_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['FTSEMIB.MI'], stock_n_ups_5, past_days=5, col_name='input_milan_n_ups_5d')
    # nordic_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^OMX'], stock_n_ups_90, past_days=90, col_name='input_nordic_n_ups_90d')
    # nordic_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^OMX'], stock_n_ups_30, past_days=30, col_name='input_nordic_n_ups_30d')
    # nordic_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^OMX'], stock_n_ups_5, past_days=5, col_name='input_nordic_n_ups_5d')
    # amsterdam_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^AEX'], stock_n_ups_90, past_days=90, col_name='input_amsterdam_n_ups_90d')
    # amsterdam_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^AEX'], stock_n_ups_30, past_days=30, col_name='input_amsterdam_n_ups_30d')
    # amsterdam_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^AEX'], stock_n_ups_5, past_days=5, col_name='input_amsterdam_n_ups_5d')
    # paris_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^FCHI'], stock_n_ups_90, past_days=90, col_name='input_paris_n_ups_90d')
    # paris_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^FCHI'], stock_n_ups_30, past_days=30, col_name='input_paris_n_ups_30d')
    # paris_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^FCHI'], stock_n_ups_5, past_days=5, col_name='input_paris_n_ups_5d')
    sp500_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^GSPC'], stock_n_ups_90, past_days=90, col_name='input_sp500_n_ups_90d')
    sp500_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^GSPC'], stock_n_ups_30, past_days=30, col_name='input_sp500_n_ups_30d')
    sp500_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^GSPC'], stock_n_ups_5, past_days=5, col_name='input_sp500_n_ups_5d')
    euro_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^STOXX50E'], stock_n_ups_90, past_days=90, col_name='input_euro_n_ups_90d')
    euro_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^STOXX50E'], stock_n_ups_30, past_days=30, col_name='input_euro_n_ups_30d')
    euro_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^STOXX50E'], stock_n_ups_5, past_days=5, col_name='input_euro_n_ups_5d')
    # frankfurt_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^GDAXI'], stock_n_ups_90, past_days=90, col_name='input_frankfurt_n_ups_90d')
    # frankfurt_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^GDAXI'], stock_n_ups_30, past_days=30, col_name='input_frankfurt_n_ups_30d')
    # frankfurt_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^GDAXI'], stock_n_ups_5, past_days=5, col_name='input_frankfurt_n_ups_5d')
    # london_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^FTSE'], stock_n_ups_90, past_days=90, col_name='input_london_n_ups_90d')
    # london_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^FTSE'], stock_n_ups_30, past_days=30, col_name='input_london_n_ups_30d')
    # london_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^FTSE'], stock_n_ups_5, past_days=5, col_name='input_london_n_ups_5d')
    vix_n_ups_90 = get_market_n_ups(dfs['df_market_buy']['^VIX'], stock_n_ups_90, past_days=90, col_name='input_vix_n_ups_90d')
    vix_n_ups_30 = get_market_n_ups(dfs['df_market_buy']['^VIX'], stock_n_ups_30, past_days=30, col_name='input_vix_n_ups_30d')
    vix_n_ups_5 = get_market_n_ups(dfs['df_market_buy']['^VIX'], stock_n_ups_5, past_days=5, col_name='input_vix_n_ups_5d')

    var_rank_90 = get_var_rank(dfs['df_stock_buy'], past_days=90)
    var_rank_30 = get_var_rank(dfs['df_stock_buy'], past_days=30)
    var_rank_10 = get_var_rank(dfs['df_stock_buy'], past_days=10)
    var_rank_1 = get_var_rank(dfs['df_stock_buy'], past_days=1)

    perf_vs_market_90 = get_performance_vs_market(dfs['df_stock_buy'], past_days=90)
    perf_vs_market_30 = get_performance_vs_market(dfs['df_stock_buy'], past_days=30)
    perf_vs_market_10 = get_performance_vs_market(dfs['df_stock_buy'], past_days=10)
    perf_vs_market_1 = get_performance_vs_market(dfs['df_stock_buy'], past_days=1)

    # rsi_14 = calculate_rsi(dfs['df_stock_buy'], period=14)
    # macd, macd_signal = calculate_macd(dfs['df_stock_buy'])
    # atr_14 = calculate_atr(dfs['df_stock_high'], dfs['df_stock_low'], dfs['df_stock_close'], period=14)
    # bollinger_upper, bollinger_lower = calculate_bollinger_bands(dfs['df_stock_buy'])

    current_ratio = hf.stack(dfs['df_current_ratio'], 'input_current_ratio')
    ev_to_ebitda_ltm =  hf.stack(dfs['df_ev_to_ebitda_ltm'], 'input_ev_to_ebitda_ltm')
    fcf_yield_ltm =  hf.stack(dfs['df_fcf_yield_ltm'], 'input_fcf_yield_ltm')
    marketcap =  hf.stack(dfs['df_marketcap'], 'input_marketcap')
    pe_ltm =  hf.stack(dfs['df_pe_ltm'], 'input_pe_ltm')
    price_to_book =  hf.stack(dfs['df_price_to_book'], 'input_price_to_book')
    roa =  hf.stack(dfs['df_roa'], 'input_roa')
    roe =  hf.stack(dfs['df_roe'], 'input_roe')
    total_debt =  hf.stack(dfs['df_total_debt'], 'input_total_debt')
    total_rev =  hf.stack(dfs['df_total_rev'], 'input_total_rev')

    current_ratio_var_1 =  hf.stack(dfs['df_current_ratio_var_1'], 'input_current_ratio_var_1')
    ev_to_ebitda_ltm_var_1 =  hf.stack(dfs['df_ev_to_ebitda_ltm_var_1'], 'input_ev_to_ebitda_ltm_var_1')
    fcf_yield_ltm_var_1 =  hf.stack(dfs['df_fcf_yield_ltm_var_1'], 'input_fcf_yield_ltm_var_1')
    marketcap_var_1 =  hf.stack(dfs['df_marketcap_var_1'], 'input_marketcap_var_1')
    pe_ltm_var_1 =  hf.stack(dfs['df_pe_ltm_var_1'], 'input_pe_ltm_var_1')
    price_to_book_var_1 =  hf.stack(dfs['df_price_to_book_var_1'], 'input_price_to_book_var_1')
    roa_var_1 =  hf.stack(dfs['df_roa_var_1'], 'input_roa_var_1')
    roe_var_1 =  hf.stack(dfs['df_roe_var_1'], 'input_roe_var_1')
    total_debt_var_1 =  hf.stack(dfs['df_total_debt_var_1'], 'input_total_debt_var_1')
    total_rev_var_1 =  hf.stack(dfs['df_total_rev_var_1'], 'input_total_rev_var_1')

    # current_ratio_var_2 =  hf.stack(dfs['df_current_ratio_var_2'], 'input_current_ratio_var_2')
    # ev_to_ebitda_ltm_var_2 =  hf.stack(dfs['df_ev_to_ebitda_ltm_var_2'], 'input_ev_to_ebitda_ltm_var_2')
    # fcf_yield_ltm_var_2 =  hf.stack(dfs['df_fcf_yield_ltm_var_2'], 'input_fcf_yield_ltm_var_2')
    # marketcap_var_2 =  hf.stack(dfs['df_marketcap_var_2'], 'input_marketcap_var_2')
    # pe_ltm_var_2 =  hf.stack(dfs['df_pe_ltm_var_2'], 'input_pe_ltm_var_2')
    # price_to_book_var_2 =  hf.stack(dfs['df_price_to_book_var_2'], 'input_price_to_book_var_2')
    # roa_var_2 =  hf.stack(dfs['df_roa_var_2'], 'input_roa_var_2')
    # roe_var_2 =  hf.stack(dfs['df_roe_var_2'], 'input_roe_var_2')
    # total_debt_var_2 =  hf.stack(dfs['df_total_debt_var_2'], 'input_total_debt_var_2')
    # total_rev_var_2 =  hf.stack(dfs['df_total_rev_var_2'], 'input_total_rev_var_2')

    # current_ratio_var_4 =  hf.stack(dfs['df_current_ratio_var_4'], 'input_current_ratio_var_4')
    # ev_to_ebitda_ltm_var_4 =  hf.stack(dfs['df_ev_to_ebitda_ltm_var_4'], 'input_ev_to_ebitda_ltm_var_4')
    # fcf_yield_ltm_var_4 =  hf.stack(dfs['df_fcf_yield_ltm_var_4'], 'input_fcf_yield_ltm_var_4')
    # marketcap_var_4 =  hf.stack(dfs['df_marketcap_var_4'], 'input_marketcap_var_4')
    # pe_ltm_var_4 =  hf.stack(dfs['df_pe_ltm_var_4'], 'input_pe_ltm_var_4')
    # price_to_book_var_4 =  hf.stack(dfs['df_price_to_book_var_4'], 'input_price_to_book_var_4')
    # roa_var_4 =  hf.stack(dfs['df_roa_var_4'], 'input_roa_var_4')
    # roe_var_4 =  hf.stack(dfs['df_roe_var_4'], 'input_roe_var_4')
    # total_debt_var_4 =  hf.stack(dfs['df_total_debt_var_4'], 'input_total_debt_var_4')
    # total_rev_var_4 =  hf.stack(dfs['df_total_rev_var_4'], 'input_total_rev_var_4')

    current_ratio_rank = get_rank(dfs['df_current_ratio'], 'input_current_ratio_rank')
    ev_to_ebitda_ltm_rank = get_rank(dfs['df_ev_to_ebitda_ltm'], 'input_ev_to_ebitda_ltm_rank')
    fcf_yield_ltm_rank = get_rank(dfs['df_fcf_yield_ltm'], 'input_fcf_yield_ltm_rank')
    marketcap_rank = get_rank(dfs['df_marketcap'], 'input_marketcap_rank')
    pe_ltm_rank = get_rank(dfs['df_pe_ltm'], 'input_pe_ltm_rank')
    price_to_book_rank = get_rank(dfs['df_price_to_book'], 'input_price_to_book_rank')
    roa_rank = get_rank(dfs['df_roa'], 'input_roa_rank')
    roe_rank = get_rank(dfs['df_roe'], 'input_roe_rank')
    total_debt_rank = get_rank(dfs['df_total_debt'], 'input_total_debt_rank')
    total_rev_rank = get_rank(dfs['df_total_rev'], 'input_total_rev_rank')

    input_list = [
        var_stock_90, var_stock_30, var_stock_10, var_stock_1,
        # var_brussels_market_90, var_brussels_market_30, var_brussels_market_10, var_brussels_market_1,
        # var_madrid_market_90, var_madrid_market_30, var_madrid_market_10, var_madrid_market_1,
        # var_milan_market_90, var_milan_market_30, var_milan_market_10, var_milan_market_1,
        # var_nordic_market_90, var_nordic_market_30, var_nordic_market_10, var_nordic_market_1,
        # var_amsterdam_market_90, var_amsterdam_market_30, var_amsterdam_market_10, var_amsterdam_market_1,
        # var_paris_market_90, var_paris_market_30, var_paris_market_10, var_paris_market_1,
        # var_sp500_market_90, var_sp500_market_30, var_sp500_market_10, var_sp500_market_1,
        # var_euro_market_90, var_euro_market_30, var_euro_market_10, var_euro_market_1,
        # var_frankfurt_market_90, var_frankfurt_market_30, var_frankfurt_market_10, var_frankfurt_market_1,
        # var_london_market_90, var_london_market_30, var_london_market_10, var_london_market_1,
        # var_vix_90, var_vix_30, var_vix_10, var_vix_1,

        # market_name,

        var_vs_stock_close_1, var_vs_stock_high_1, var_vs_stock_low_1,
        # var_vs_brussels_close_1, var_vs_brussels_high_1, var_vs_brussels_low_1,
        # var_vs_madrid_close_1, var_vs_madrid_high_1, var_vs_madrid_low_1,
        # var_vs_milan_close_1, var_vs_milan_high_1, var_vs_milan_low_1,
        # var_vs_nordic_close_1, var_vs_nordic_high_1, var_vs_nordic_low_1,
        # var_vs_amsterdam_close_1, var_vs_amsterdam_high_1, var_vs_amsterdam_low_1,
        # var_vs_paris_close_1, var_vs_paris_high_1, var_vs_paris_low_1,
        var_vs_sp500_close_1, var_vs_sp500_high_1, var_vs_sp500_low_1,
        var_vs_euro_close_1, var_vs_euro_high_1, var_vs_euro_low_1,
        # var_vs_frankfurt_close_1, var_vs_frankfurt_high_1, var_vs_frankfurt_low_1,
        # var_vs_london_close_1, var_vs_london_high_1, var_vs_london_low_1,
        var_vs_vix_close_1, var_vs_vix_high_1, var_vs_vix_low_1,
        
        # volume_var_90_1, volume_var_30_1, volume_var_10_1, volume_var_2_1,
        # min_var_90, min_var_30, min_var_10,
        # max_var_90, max_var_30, max_var_10,
        days_since_min_30, days_since_min_10,
        days_since_max_30, days_since_max_10,
        volatility_30, volatility_10, volatility_2,
        # market_volatility_30, market_volatility_10, market_volatility_2,
        volume_volability_90_1, volume_volability_30_1, volume_volability_10_1, volume_volability_2_1,
        
        stock_n_ups_90, stock_n_ups_30, stock_n_ups_5,
        # brussels_n_ups_90, brussels_n_ups_30, brussels_n_ups_5,
        # madrid_n_ups_90, madrid_n_ups_30, madrid_n_ups_5,
        # milan_n_ups_90, milan_n_ups_30, milan_n_ups_5,
        # nordic_n_ups_90, nordic_n_ups_30, nordic_n_ups_5,
        # amsterdam_n_ups_90, amsterdam_n_ups_30, amsterdam_n_ups_5,
        # paris_n_ups_90, paris_n_ups_30, paris_n_ups_5,
        sp500_n_ups_90, sp500_n_ups_30, sp500_n_ups_5,
        euro_n_ups_90, euro_n_ups_30, euro_n_ups_5,
        # frankfurt_n_ups_90, frankfurt_n_ups_30, frankfurt_n_ups_5,
        # london_n_ups_90, london_n_ups_30, london_n_ups_5,
        vix_n_ups_90, vix_n_ups_30, vix_n_ups_5,

        var_rank_90, var_rank_30, var_rank_10, var_rank_1,
        perf_vs_market_90, perf_vs_market_30, perf_vs_market_10, perf_vs_market_1,
        # rsi_14, macd, macd_signal,
        # atr_14, bollinger_upper, bollinger_lower, # affected by stock absolute prices
        
        current_ratio, ev_to_ebitda_ltm, fcf_yield_ltm, marketcap, pe_ltm, price_to_book, roa, roe, total_debt, total_rev,
        current_ratio_var_1, ev_to_ebitda_ltm_var_1, fcf_yield_ltm_var_1, marketcap_var_1, pe_ltm_var_1, price_to_book_var_1, roa_var_1, roe_var_1, total_debt_var_1, total_rev_var_1,
        # current_ratio_var_2, ev_to_ebitda_ltm_var_2, fcf_yield_ltm_var_2, marketcap_var_2, pe_ltm_var_2, price_to_book_var_2, roa_var_2, roe_var_2, total_debt_var_2, total_rev_var_2,
        # current_ratio_var_4, ev_to_ebitda_ltm_var_4, fcf_yield_ltm_var_4, marketcap_var_4, pe_ltm_var_4, price_to_book_var_4, roa_var_4, roe_var_4, total_debt_var_4, total_rev_var_4,
        current_ratio_rank, ev_to_ebitda_ltm_rank, fcf_yield_ltm_rank, marketcap_rank, pe_ltm_rank, price_to_book_rank, roa_rank, roe_rank, total_debt_rank, total_rev_rank
    ]
    
    if buying_time == 'Close':
        var_vs_open_0 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_open'], past_days=0, col_name='input_var_vs_open_0d')
        var_vs_low_0 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_low'], past_days=0, col_name='input_var_vs_low_0d')
        var_vs_high_0 = calculate_stock_var_vs_past_ohlcv(dfs['df_stock_buy'], dfs['df_stock_high'], past_days=0, col_name='input_var_vs_high_0d')

        input_list += [var_vs_open_0, var_vs_low_0, var_vs_high_0]

    print(f'Number of inputs: {len(input_list)}')

    df_inputs = pd.concat(input_list, axis='columns')

    return df_inputs