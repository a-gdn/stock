import utils.helper_functions as hf
import config as cfg

import pandas as pd
import numpy as np

def calculate_rsi(df, period=14):
    hf.validate_dataframe(df, function_name="calculate_rsi")

    rsi = hf.calculate_rsi(df, period)
    return hf.stack(rsi, f'input_rsi_{period}d')

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    hf.validate_dataframe(df, function_name="calculate_macd")

    macd, signal = hf.calculate_macd(df, short_period, long_period, signal_period)
    return hf.stack(macd, 'input_macd'), hf.stack(signal, 'input_macd_signal')

def calculate_atr(df_high, df_low, df_close, period=14):
    hf.validate_dataframe(df_high, function_name="calculate_atr - df_high")
    hf.validate_dataframe(df_low, function_name="calculate_atr - df_low")
    hf.validate_dataframe(df_close, function_name="calculate_atr - df_close")

    atr = hf.calculate_atr(df_high, df_low, df_close, period)
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


def calculate_var(df, past_days, col_name):
    var = hf.calculate_variations(df, past_days, n_future_days=0)
    var_stacked = hf.stack(var, col_name)

    return var_stacked

def calculate_var_vs_past_ohlcv(df, df_past, past_days, col_name):
    var = df / df_past.shift(past_days)
    var_stacked = hf.stack(var, col_name)

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

def get_volume_volability(df, past_start_day, past_end_day):
    df_shifted = df.shift(past_end_day)
    window_size = past_start_day - past_end_day + 1
    volatility = hf.calculate_volatility(df_shifted, window_size)
    volatility_stacked = hf.stack(volatility, f'input_volume_volatility_{past_start_day}-{past_end_day}d')

    return volatility_stacked

def get_n_ups(df, past_days):
    n_ups = hf.calculate_n_ups(df, past_days)
    n_ups_stacked = hf.stack(n_ups, f'input_n_ups_{past_days}d')

    return n_ups_stacked

def get_performance_vs_market(df, past_days):
    performance_vs_market = hf.calculate_performance_vs_market(df, past_days)
    performance_vs_market_stacked = hf.stack(performance_vs_market, f'input_perf_vs_market_{past_days}d')

    return performance_vs_market_stacked

def get_rank(df, past_days):
    rank = hf.calculate_rank(df, past_days, n_future_days=0)
        
    return hf.stack(rank, f'input_rank_{past_days}d')

def format_ref(df, col_name, df_index):
    df = hf.ensure_dataframe(df)
    df = hf.rename_first_column(df, new_col_name=col_name)
    df_reindexed = df_index.join(df, on='Date', how='left') # Apply index from another df

    return df_reindexed

def calculate_ref_var(df, past_days, col_name, df_index):
    var = hf.calculate_variations(df, past_days, n_future_days=0)
    var_reindexed = format_ref(var, col_name, df_index)

    return var_reindexed

def get_inputs(df_buy, dfs_ohlcv, buying_time):
    var_90 = calculate_var(df_buy, past_days=90, col_name='input_var_90d')
    var_30 = calculate_var(df_buy, past_days=30, col_name='input_var_30d')
    var_10 = calculate_var(df_buy, past_days=10, col_name='input_var_10d')
    var_1 = calculate_var(df_buy, past_days=1, col_name='input_var_1d')

    # sp500_var_90 = calculate_ref_var(df_sp500['Close'], past_days=90, col_name='input_sp500_var_90d', df_index=var_1)
    # sp500_var_30 = calculate_ref_var(df_sp500['Close'], past_days=30, col_name='input_sp500_var_30d', df_index=var_1)
    # sp500_var_10 = calculate_ref_var(df_sp500['Close'], past_days=10, col_name='input_sp500_var_10d', df_index=var_1)
    # sp500_var_1 = calculate_ref_var(df_sp500['Close'], past_days=1, col_name='input_sp500_var_1d', df_index=var_1)
    # vix = format_ref(df_vix['Close'], col_name='input_vix', df_index=var_1)

    var_vs_close_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_close'], past_days=1, col_name='input_var_vs_close_1d')
    var_vs_low_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_low'], past_days=1, col_name='input_var_vs_low_1d')
    var_vs_high_1 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_high'], past_days=1, col_name='input_var_vs_high_1d')

    # volume_1 = get_volume(dfs_ohlcv['df_volume'], past_day=1)
    
    volume_var_90_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=90, past_end_day=1)
    volume_var_30_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=30, past_end_day=1)
    volume_var_10_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=10, past_end_day=1)
    volume_var_2_1 = calculate_volume_var(dfs_ohlcv['df_volume'], past_start_day=2, past_end_day=1)
    
    # market_var_90 = calculate_market_var(df_buy, past_days=90)
    # market_var_30 = calculate_market_var(df_buy, past_days=30)
    # market_var_10 = calculate_market_var(df_buy, past_days=10)
    # market_var_5 = calculate_market_var(df_buy, past_days=5)
    # market_var_1 = calculate_market_var(df_buy, past_days=1)
    
    min_var_90, max_var_90 = min_max_var(df_buy, past_days=90)
    min_var_30, max_var_30 = min_max_var(df_buy, past_days=30)
    min_var_10, max_var_10 = min_max_var(df_buy, past_days=10)

    days_since_min_30, days_since_max_30 = days_since_min_max(df_buy, past_days=30)
    days_since_min_10, days_since_max_10 = days_since_min_max(df_buy, past_days=10)

    volatility_30 = get_volatility(df_buy, past_days=30)
    volatility_10 = get_volatility(df_buy, past_days=10)
    volatility_2 = get_volatility(df_buy, past_days=2)

    # market_volatility_30 = get_market_volatility(df_buy, past_days=30)
    # market_volatility_10 = get_market_volatility(df_buy, past_days=10)
    # market_volatility_2 = get_market_volatility(df_buy, past_days=2)

    volume_volability_90_1 = get_volume_volability(dfs_ohlcv['df_volume'], past_start_day=90, past_end_day=1)
    volume_volability_30_1 = get_volume_volability(dfs_ohlcv['df_volume'], past_start_day=30, past_end_day=1)
    volume_volability_10_1 = get_volume_volability(dfs_ohlcv['df_volume'], past_start_day=10, past_end_day=1)
    volume_volability_2_1 = get_volume_volability(dfs_ohlcv['df_volume'], past_start_day=2, past_end_day=1)

    n_ups_90 = get_n_ups(df_buy, past_days=90)
    n_ups_30 = get_n_ups(df_buy, past_days=30)
    n_ups_5 = get_n_ups(df_buy, past_days=5)

    rank_90 = get_rank(df_buy, past_days=90)
    rank_30 = get_rank(df_buy, past_days=30)
    rank_10 = get_rank(df_buy, past_days=10)
    rank_1 = get_rank(df_buy, past_days=1)

    perf_vs_market_90 = get_performance_vs_market(df_buy, past_days=90)
    perf_vs_market_30 = get_performance_vs_market(df_buy, past_days=30)
    perf_vs_market_10 = get_performance_vs_market(df_buy, past_days=10)
    perf_vs_market_1 = get_performance_vs_market(df_buy, past_days=1)

    # rsi_14 = calculate_rsi(df_buy, period=14)
    # macd, macd_signal = calculate_macd(df_buy)
    # atr_14 = calculate_atr(dfs_ohlcv['df_high'], dfs_ohlcv['df_low'], dfs_ohlcv['df_close'], period=14)
    # bollinger_upper, bollinger_lower = calculate_bollinger_bands(df_buy)

    current_ratio = hf.stack(dfs_ohlcv['df_current_ratio'], 'input_current_ratio')
    ev_to_ebitda_ltm =  hf.stack(dfs_ohlcv['df_ev_to_ebitda_ltm'], 'input_ev_to_ebitda_ltm')
    fcf_yield_ltm =  hf.stack(dfs_ohlcv['df_fcf_yield_ltm'], 'input_fcf_yield_ltm')
    marketcap =  hf.stack(dfs_ohlcv['df_marketcap'], 'input_marketcap')
    pe_ltm =  hf.stack(dfs_ohlcv['df_pe_ltm'], 'input_pe_ltm')
    price_to_book =  hf.stack(dfs_ohlcv['df_price_to_book'], 'input_price_to_book')
    roa =  hf.stack(dfs_ohlcv['df_roa'], 'input_roa')
    roe =  hf.stack(dfs_ohlcv['df_roe'], 'input_roe')
    total_debt =  hf.stack(dfs_ohlcv['df_total_debt'], 'input_total_debt')
    total_rev =  hf.stack(dfs_ohlcv['df_total_rev'], 'input_total_rev')

    current_ratio_var_1 =  hf.stack(dfs_ohlcv['df_current_ratio_var_1'], 'input_current_ratio_var_1')
    ev_to_ebitda_ltm_var_1 =  hf.stack(dfs_ohlcv['df_ev_to_ebitda_ltm_var_1'], 'input_ev_to_ebitda_ltm_var_1')
    fcf_yield_ltm_var_1 =  hf.stack(dfs_ohlcv['df_fcf_yield_ltm_var_1'], 'input_fcf_yield_ltm_var_1')
    marketcap_var_1 =  hf.stack(dfs_ohlcv['df_marketcap_var_1'], 'input_marketcap_var_1')
    pe_ltm_var_1 =  hf.stack(dfs_ohlcv['df_pe_ltm_var_1'], 'input_pe_ltm_var_1')
    price_to_book_var_1 =  hf.stack(dfs_ohlcv['df_price_to_book_var_1'], 'input_price_to_book_var_1')
    roa_var_1 =  hf.stack(dfs_ohlcv['df_roa_var_1'], 'input_roa_var_1')
    roe_var_1 =  hf.stack(dfs_ohlcv['df_roe_var_1'], 'input_roe_var_1')
    total_debt_var_1 =  hf.stack(dfs_ohlcv['df_total_debt_var_1'], 'input_total_debt_var_1')
    total_rev_var_1 =  hf.stack(dfs_ohlcv['df_total_rev_var_1'], 'input_total_rev_var_1')

    current_ratio_var_2 =  hf.stack(dfs_ohlcv['df_current_ratio_var_2'], 'input_current_ratio_var_2')
    ev_to_ebitda_ltm_var_2 =  hf.stack(dfs_ohlcv['df_ev_to_ebitda_ltm_var_2'], 'input_ev_to_ebitda_ltm_var_2')
    fcf_yield_ltm_var_2 =  hf.stack(dfs_ohlcv['df_fcf_yield_ltm_var_2'], 'input_fcf_yield_ltm_var_2')
    marketcap_var_2 =  hf.stack(dfs_ohlcv['df_marketcap_var_2'], 'input_marketcap_var_2')
    pe_ltm_var_2 =  hf.stack(dfs_ohlcv['df_pe_ltm_var_2'], 'input_pe_ltm_var_2')
    price_to_book_var_2 =  hf.stack(dfs_ohlcv['df_price_to_book_var_2'], 'input_price_to_book_var_2')
    roa_var_2 =  hf.stack(dfs_ohlcv['df_roa_var_2'], 'input_roa_var_2')
    roe_var_2 =  hf.stack(dfs_ohlcv['df_roe_var_2'], 'input_roe_var_2')
    total_debt_var_2 =  hf.stack(dfs_ohlcv['df_total_debt_var_2'], 'input_total_debt_var_2')
    total_rev_var_2 =  hf.stack(dfs_ohlcv['df_total_rev_var_2'], 'input_total_rev_var_2')

    current_ratio_var_4 =  hf.stack(dfs_ohlcv['df_current_ratio_var_4'], 'input_current_ratio_var_4')
    ev_to_ebitda_ltm_var_4 =  hf.stack(dfs_ohlcv['df_ev_to_ebitda_ltm_var_4'], 'input_ev_to_ebitda_ltm_var_4')
    fcf_yield_ltm_var_4 =  hf.stack(dfs_ohlcv['df_fcf_yield_ltm_var_4'], 'input_fcf_yield_ltm_var_4')
    marketcap_var_4 =  hf.stack(dfs_ohlcv['df_marketcap_var_4'], 'input_marketcap_var_4')
    pe_ltm_var_4 =  hf.stack(dfs_ohlcv['df_pe_ltm_var_4'], 'input_pe_ltm_var_4')
    price_to_book_var_4 =  hf.stack(dfs_ohlcv['df_price_to_book_var_4'], 'input_price_to_book_var_4')
    roa_var_4 =  hf.stack(dfs_ohlcv['df_roa_var_4'], 'input_roa_var_4')
    roe_var_4 =  hf.stack(dfs_ohlcv['df_roe_var_4'], 'input_roe_var_4')
    total_debt_var_4 =  hf.stack(dfs_ohlcv['df_total_debt_var_4'], 'input_total_debt_var_4')
    total_rev_var_4 =  hf.stack(dfs_ohlcv['df_total_rev_var_4'], 'input_total_rev_var_4')

    input_list = [
        var_90, var_30, var_10, var_1,
        var_vs_close_1, var_vs_high_1, var_vs_low_1,
        # volume_1,
        volume_var_90_1, volume_var_30_1, volume_var_10_1, volume_var_2_1,
        # market_var_90, market_var_30, market_var_10, market_var_5, market_var_1,
        min_var_90, min_var_30, min_var_10,
        max_var_90, max_var_30, max_var_10,
        days_since_min_30, days_since_min_10,
        days_since_max_30, days_since_max_10,
        volatility_30, volatility_10, volatility_2,
        # market_volatility_30, market_volatility_10, market_volatility_2,
        volume_volability_90_1, volume_volability_30_1, volume_volability_10_1, volume_volability_2_1,
        n_ups_90, n_ups_30, n_ups_5,
        rank_90, rank_30, rank_10, rank_1,
        perf_vs_market_90, perf_vs_market_30, perf_vs_market_10, perf_vs_market_1,
        # rsi_14, macd, macd_signal,
        # atr_14, bollinger_upper, bollinger_lower, # affected by stock absolute prices
        # sp500_var_90, sp500_var_30, sp500_var_10, sp500_var_1,
        # vix
        current_ratio, ev_to_ebitda_ltm, fcf_yield_ltm, marketcap, pe_ltm, price_to_book, roa, roe, total_debt, total_rev,
        current_ratio_var_1, ev_to_ebitda_ltm_var_1, fcf_yield_ltm_var_1, marketcap_var_1, pe_ltm_var_1, price_to_book_var_1, roa_var_1, roe_var_1, total_debt_var_1, total_rev_var_1,
        current_ratio_var_2, ev_to_ebitda_ltm_var_2, fcf_yield_ltm_var_2, marketcap_var_2, pe_ltm_var_2, price_to_book_var_2, roa_var_2, roe_var_2, total_debt_var_2, total_rev_var_2,
        current_ratio_var_4, ev_to_ebitda_ltm_var_4, fcf_yield_ltm_var_4, marketcap_var_4, pe_ltm_var_4, price_to_book_var_4, roa_var_4, roe_var_4, total_debt_var_4, total_rev_var_4
    ]
    
    if buying_time == 'Close':
        var_vs_open_0 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_open'], past_days=0, col_name='input_var_vs_open_0d')
        var_vs_low_0 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_low'], past_days=0, col_name='input_var_vs_low_0d')
        var_vs_high_0 = calculate_var_vs_past_ohlcv(df_buy, dfs_ohlcv['df_high'], past_days=0, col_name='input_var_vs_high_0d')

        input_list += [var_vs_open_0, var_vs_low_0, var_vs_high_0]

    df_inputs = pd.concat(input_list, axis='columns')
    # df_data = df_data.dropna()

    return df_inputs