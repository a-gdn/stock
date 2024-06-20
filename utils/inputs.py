import utils.helper_functions as hf

import pandas as pd
import numpy as np

def calculate_var(df, past_days, future_days):
    var = hf.calculate_variations(df, past_days, future_days)
    var_stacked = hf.stack(var, f'input_var_past_{past_days}d_future_{future_days}d')

    return var_stacked

def calculate_var_vs_past_ohlcv(df, df_past, past_days, title):
    var = df / df_past.shift(past_days)
    var_stacked = hf.stack(var, f'input_var_past_{title}_{past_days}d')

    return var_stacked

def clean_volume_data(df_volume):
    # replaces 0 and infinity values by np.nan
    df_volume.replace(0, np.nan, inplace=True)
    df_volume.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df_volume

def get_volume(df_volume, past_day):
    df_cleaned_volume = clean_volume_data(df_volume)
    volume_var = df_cleaned_volume.shift(past_day)

    volume_var_stacked = hf.stack(volume_var, f'input_volume_{past_day}d')

    return volume_var_stacked

def calculate_volume_var(df_volume, past_start_day, past_end_day):
    df_cleaned_volume = clean_volume_data(df_volume)

    volume_var = df_cleaned_volume.shift(past_end_day) / df_cleaned_volume.shift(past_start_day)
    volume_var_stacked = hf.stack(volume_var, f'input_volume_var_{past_start_day}-{past_end_day}d')

    return volume_var_stacked

def calculate_market_var(df, past_days):
    market_var = hf.calculate_market_variations(df, past_days)
    market_var_stacked = hf.stack(market_var, f'input_market_var_{past_days}d')

    return market_var_stacked

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

def get_volume_volability(df, past_days):
    volatility = hf.calculate_volatility(df, past_days)
    volatility_stacked = hf.stack(volatility, f'input_volume_volatility_{past_days}d')

    return volatility_stacked

def get_n_ups(df, past_days):
    n_ups = hf.calculate_n_ups(df, past_days)
    n_ups_stacked = hf.stack(n_ups, f'input_n_ups_{past_days}d')

    return n_ups_stacked

def get_performance_vs_market(df, past_days):
    performance_vs_market = hf.calculate_performance_vs_market(df, past_days)
    performance_vs_market_stacked = hf.stack(performance_vs_market, f'input_perf_vs_market_{past_days}d')

    return performance_vs_market_stacked

def get_rank(df, past_days, future_days):
    rank = hf.calculate_rank(df, past_days, future_days)
    
    if future_days == 0:
        rank_stacked = hf.stack(rank, f'input_rank_{past_days}d')
    elif past_days == 0:
        rank_stacked = hf.stack(rank, f'output_rank_{future_days}d')
    else:
        raise ValueError('Either past_days or future_days must be 0')
    
    return rank_stacked

def get_inputs(df_buy, dfs_ohlcv):
    var_90 = calculate_var(df_buy, past_days=90, future_days=0)
    var_60 = calculate_var(df_buy, past_days=60, future_days=0)
    var_30 = calculate_var(df_buy, past_days=30, future_days=0)
    var_10 = calculate_var(df_buy, past_days=10, future_days=0)
    var_5 = calculate_var(df_buy, past_days=5, future_days=0)
    var_2 = calculate_var(df_buy, past_days=2, future_days=0)
    var_1 = calculate_var(df_buy, past_days=1, future_days=0)

    var_vs_close_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_close'], past_days=1, title='close')
    var_vs_low_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_low'], past_days=1, title='low')
    var_vs_high_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_high'], past_days=1, title='high')

    # volume_1 = get_volume(dfs_ohlcv['df_volume'], past_day=1)
    
    volume_var_90_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=90, past_end_day=1)
    volume_var_60_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=60, past_end_day=1)
    volume_var_30_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=30, past_end_day=1)
    volume_var_10_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=10, past_end_day=1)
    volume_var_3_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=3, past_end_day=1)
    volume_var_2_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=2, past_end_day=1)
    
    # market_var_90 = calculate_market_var(df_buy, past_days=90)
    # market_var_30 = calculate_market_var(df_buy, past_days=30)
    # market_var_10 = calculate_market_var(df_buy, past_days=10)
    # market_var_5 = calculate_market_var(df_buy, past_days=5)
    # market_var_1 = calculate_market_var(df_buy, past_days=1)
    
    min_var_90, max_var_90 = min_max_var(df_buy, past_days=90)
    min_var_30, max_var_30 = min_max_var(df_buy, past_days=30)
    min_var_10, max_var_10 = min_max_var(df_buy, past_days=10)
    min_var_5, max_var_5 = min_max_var(df_buy, past_days=5)
    min_var_2, max_var_2 = min_max_var(df_buy, past_days=2)

    days_since_min_30, days_since_max_30 = days_since_min_max(df_buy, past_days=30)
    days_since_min_10, days_since_max_10 = days_since_min_max(df_buy, past_days=10)

    volatility_30 = get_volatility(df_buy, past_days=30)
    volatility_10 = get_volatility(df_buy, past_days=10)
    volatility_2 = get_volatility(df_buy, past_days=2)

    # market_volatility_30 = get_market_volatility(df_buy, past_days=30)
    # market_volatility_10 = get_market_volatility(df_buy, past_days=10)
    # market_volatility_2 = get_market_volatility(df_buy, past_days=2)

    volume_volability_90 = get_volume_volability(dfs_ohlcv['df_volume'], past_days=90)
    volume_volability_30 = get_volume_volability(dfs_ohlcv['df_volume'], past_days=30)
    volume_volability_10 = get_volume_volability(dfs_ohlcv['df_volume'], past_days=10)
    volume_volability_2 = get_volume_volability(dfs_ohlcv['df_volume'], past_days=2)

    n_ups_90 = get_n_ups(df_buy, past_days=90)
    n_ups_30 = get_n_ups(df_buy, past_days=30)
    n_ups_5 = get_n_ups(df_buy, past_days=5)

    rank_90 = get_rank(df_buy, past_days=90, future_days=0)
    rank_30 = get_rank(df_buy, past_days=30, future_days=0)
    rank_10 = get_rank(df_buy, past_days=10, future_days=0)
    rank_5 = get_rank(df_buy, past_days=5, future_days=0)
    rank_2 = get_rank(df_buy, past_days=2, future_days=0)
    rank_1 = get_rank(df_buy, past_days=1, future_days=0)

    perf_vs_market_90 = get_performance_vs_market(df_buy, past_days=90)
    perf_vs_market_30 = get_performance_vs_market(df_buy, past_days=30)
    perf_vs_market_10 = get_performance_vs_market(df_buy, past_days=10)
    perf_vs_market_5 = get_performance_vs_market(df_buy, past_days=5)
    perf_vs_market_2 = get_performance_vs_market(df_buy, past_days=2)
    perf_vs_market_1 = get_performance_vs_market(df_buy, past_days=1)

    df_data = pd.concat([
            var_90, var_60, var_30, var_10, var_5, var_2, var_1,
            var_vs_close_1, var_vs_high_1, var_vs_low_1,
            # volume_1,
            volume_var_90_1, volume_var_60_1, volume_var_30_1, volume_var_10_1, volume_var_2_1, volume_var_3_1,
            # market_var_90, market_var_30, market_var_10, market_var_5, market_var_1,
            min_var_90, min_var_30, min_var_10, min_var_5, min_var_2,
            max_var_90, max_var_30, max_var_10, max_var_5, max_var_2,
            days_since_min_30, days_since_min_10,
            days_since_max_30, days_since_max_10,
            volatility_30, volatility_10, volatility_2,
            # market_volatility_30, market_volatility_10, market_volatility_2,
            volume_volability_90, volume_volability_30, volume_volability_10, volume_volability_2,
            n_ups_90, n_ups_30, n_ups_5,
            rank_90, rank_30, rank_10, rank_5, rank_2, rank_1,
            perf_vs_market_90, perf_vs_market_30, perf_vs_market_10, perf_vs_market_5,
            perf_vs_market_2, perf_vs_market_1
        ], axis='columns')

    # df_data = df_data.dropna()

    return df_data