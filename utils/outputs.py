import utils.helper_functions as hf
import config as cfg

import pandas as pd
import numpy as np
import logging

def get_has_min_max_values(**hyperparams):
    buying_time = hyperparams.get('buying_time')
    selling_time = hyperparams.get('selling_time')
    target_future_days = hyperparams.get('target_future_days')

    return not (buying_time == 'Close' and selling_time == 'Open' and target_future_days == 1)

def get_future_end_var(dfs, future_days):
    df_future_end = dfs['df_stock_sell'].shift(-future_days)
    future_end_var =  df_future_end / dfs['df_stock_buy']
    future_end_var_stacked = hf.stack(future_end_var, f'output_future_end_var')
    
    return future_end_var_stacked

def get_future_max_var(dfs, future_days, buying_time, selling_time):
    future_rolling_max = hf.get_future_rolling_max(dfs['df_stock_high'], future_days, buying_time, selling_time)
    future_max_var = future_rolling_max / dfs['df_stock_buy']
    future_max_var_stacked = hf.stack(future_max_var, f'output_future_max_var')
    
    return future_max_var_stacked

def get_future_min_var(dfs, df_loss_prices, future_days, buying_time, selling_time):
    future_rolling_min = hf.get_future_rolling_min(dfs['df_stock_low'], future_days, buying_time, selling_time)
    future_min_var = future_rolling_min / dfs['df_stock_buy']
    future_min_var_stacked = hf.stack(future_min_var, f'output_future_min_var')
    
    return future_min_var_stacked

def get_future_min_var_before_max(dfs, future_days, buying_time, selling_time):
    rolling_max_positions = hf.get_future_rolling_max_position(dfs['df_stock_sell'], future_days, buying_time, selling_time)

    df_low = df_low.reset_index(drop=True)
    rolling_min = df_low.apply(lambda col: col.index.map(
            lambda row: hf.get_future_rolling_min_value(row, dfs['df_stock_low'].columns.get_loc(col.name), dfs['df_stock_low'], rolling_max_positions)
        ))
    rolling_min.index = dfs['df_stock_buy'].index
    
    var = rolling_min / dfs['df_stock_buy']
    var_stacked = hf.stack(var, f'output_future_min_var')

    return var_stacked

def get_loss_limit_prices(dfs, **hyperparams):
    loss_limit = hyperparams.get('loss_limit')
    df_loss_limit_price = loss_limit * dfs['df_stock_buy']

    return df_loss_limit_price

def get_loss_prices(dfs, df_loss_limit_price):
    df_loss_prices = dfs['df_stock_open'].where(
        dfs['df_stock_open'] <= df_loss_limit_price, # Condition 1: During the night, loss_limit is reached, ends up selling at open price
        other=df_loss_limit_price.where(
            dfs['df_stock_low'] <= df_loss_limit_price, # Condition 2: During the day, loss_limit is reached, really sells at loss_limit_price
            other=dfs['df_stock_low'] # Condition 3: loss_limit is not reached
        )
    )

    return df_loss_prices

def add_future_vars(df_data, dfs, df_loss_prices, has_min_max_values, **hyperparams):
    buying_time = hyperparams.get('buying_time')
    selling_time = hyperparams.get('selling_time')
    target_future_days = hyperparams.get('target_future_days')
    sell_at_target = hyperparams.get('sell_at_target')
    
    future_end_var = get_future_end_var(dfs, target_future_days)
        
    if has_min_max_values:
        future_max_var = get_future_max_var(dfs, target_future_days, buying_time, selling_time)
        future_min_var = get_future_min_var(dfs, df_loss_prices, target_future_days, buying_time, selling_time)
    else:
        future_max_var = future_end_var.copy()
        future_max_var.rename(columns={'output_future_end_var': 'output_future_max_var'}, inplace=True)

        future_min_var = future_end_var.copy()
        future_min_var.rename(columns={'output_future_end_var': 'output_future_min_var'}, inplace=True)
    
    if sell_at_target:
        future_min_var = get_future_min_var_before_max(dfs['df_stock_buy'], dfs['df_stock_sell'], df_loss_prices, target_future_days, buying_time, selling_time)

    future_vars = pd.concat([df_data, future_end_var, future_max_var, future_min_var], axis='columns')
    
    return future_vars

def add_output_is_loss_limit_reached(df, **hyperparams):
    loss_limit = hyperparams.get('loss_limit')
    df['output_is_loss_limit_reached'] = (df['output_future_min_var'] <= loss_limit)

    return df

def add_output_var_binary(df_data, **hyperparams):
    sell_at_target = hyperparams.get('sell_at_target')
    threshold = hyperparams.get('var_threshold')

    if sell_at_target:
        df_data['output_var_binary'] = (df_data['output_future_max_var'] > threshold).astype(int)
    else:
        df_data['output_var_binary'] = (df_data['output_future_end_var'] > threshold).astype(int)

    df_data.loc[df_data['output_is_loss_limit_reached'], 'output_var_binary'] = 0
    
    return df_data

def add_future_rank(df_data, dfs, **hyperparams):
    target_future_days = hyperparams.get('target_future_days')

    rank = hf.calculate_rank(dfs['df_stock_buy'], n_past_days=0, n_future_days=target_future_days)
    rank_stacked = hf.stack(rank, 'output_future_end_rank')

    df_data['output_future_end_rank'] = rank_stacked
    
    return df_data

def add_output_rank_binary(df_data, num_stocks, **hyperparams):
    rank_pct_threshold = hyperparams.get('rank_pct_threshold')
    rank_threshold = int(np.floor(rank_pct_threshold * num_stocks))
    
    df_data['output_rank_binary'] = (df_data['output_future_end_rank'] <= rank_threshold).astype(int)
    
    return df_data

def add_output_is_buy(df):
    df['output_is_buy'] = df[cfg.output_binary_name].eq(1)
    
    return df

def add_output_profit(df, **hyperparams):
    var_threshold = hyperparams.get('var_threshold')
    sell_at_target = hyperparams.get('sell_at_target')

    loss_condition = df['output_is_loss_limit_reached']
    reached_target_condition = sell_at_target & (df['output_future_max_var'] > var_threshold)

    df['output_profit'] = np.select(
        [
            loss_condition,  # Condition for buy and loss condition
            reached_target_condition  # Condition for buy and reached target condition
        ],
        [
            df['output_future_min_var'],  # Value if buy and meets loss condition
            var_threshold  # Value if buy and meets target condition
        ],
        default=df['output_future_end_var']  # Default value for buy condition not meeting the above
    )

    fee_coef = hf.get_fee_coef(cfg.fee)
    df['output_profit'] *= fee_coef
    
    return df

def add_outputs(df_data, dfs, num_stocks, **hyperparams):
    df_loss_limit_prices = get_loss_limit_prices(dfs, **hyperparams)
    df_loss_prices = get_loss_prices(dfs, df_loss_limit_prices)
    has_min_max_values = get_has_min_max_values(**hyperparams)

    df_data = add_future_vars(df_data, dfs, df_loss_prices, has_min_max_values, **hyperparams)
    df_data = add_output_is_loss_limit_reached(df_data, **hyperparams)
    df_data = add_output_var_binary(df_data, **hyperparams)

    df_data = add_future_rank(df_data, dfs, **hyperparams)
    df_data = add_output_rank_binary(df_data, num_stocks, **hyperparams)

    df_data = add_output_is_buy(df_data)
    df_data = add_output_profit(df_data, **hyperparams)

    return df_data