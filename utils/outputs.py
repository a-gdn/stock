import utils.helper_functions as hf
import utils.inputs as inputs

import pandas as pd
import numpy as np

def get_future_end_var(df_buy, df_sell, future_days):
    df_future_end = df_sell.shift(-future_days)
    future_end_var =  df_future_end / df_buy
    future_end_var_stacked = hf.stack(future_end_var, f'output_future_end_var')
    
    return future_end_var_stacked

def get_future_max_var(df_buy, df_sell, future_days):
    future_rolling_max = hf.get_future_rolling_max(df_sell, future_days+1)
    future_max_var = future_rolling_max / df_buy
    future_max_var_stacked = hf.stack(future_max_var, f'output_future_max_var')
        
    return future_max_var_stacked

def get_future_min_var(df_buy, df_low, future_days):
    future_rolling_min = hf.get_future_rolling_min(df_low, future_days+1)
    future_min_var = future_rolling_min / df_buy
    future_min_var_stacked = hf.stack(future_min_var, f'output_future_min_var')
        
    return future_min_var_stacked

def get_future_min_var_before_max(df_buy, df_sell, df_low, future_days):
    rolling_max_positions = hf.get_future_rolling_max_position(df_sell, future_days)

    df_low = df_low.reset_index(drop=True)
    rolling_min = df_low.apply(lambda col: col.index.map(
            lambda row: hf.get_future_rolling_min_value(row, df_low.columns.get_loc(col.name), df_low, rolling_max_positions)
        ))
    rolling_min.index = df_buy.index
    
    var = rolling_min / df_buy
    var_stacked = hf.stack(var, f'output_future_min_var_before_max')

    return var_stacked

def classify_var(df_var, thresholds, col_name):
    df_thresholds = hf.classify_var(df_var, thresholds)

    df_thresholds_stacked = hf.stack(df_thresholds, col_name)
    df_thresholds_stacked = df_thresholds_stacked.droplevel(level=-1)

    return df_thresholds_stacked

def classify_rank(df_rank, thresholds, col_name):
    df_thresholds = hf.classify_rank(df_rank, thresholds)

    df_thresholds_stacked = hf.stack(df_thresholds, col_name)
    df_thresholds_stacked = df_thresholds_stacked.droplevel(level=-1)

    return df_thresholds_stacked

def add_future_vars(df_data, df_buy, df_sell, dfs_ohlcv, **hyperparams):
    target_future_days = hyperparams.get('target_future_days')
    sell_at_target = hyperparams.get('sell_at_target')
    
    future_end_var = get_future_end_var(df_buy, df_sell, target_future_days)
    future_max_var = get_future_max_var(df_buy, df_sell, target_future_days)
    future_min_var = get_future_min_var(df_buy, dfs_ohlcv['df_low'], target_future_days-1)
    
    df_data = pd.concat([df_data, future_end_var, future_max_var, future_min_var], axis='columns')
    
    if sell_at_target:
        future_min_var_before_max = get_future_min_var_before_max(df_buy, df_sell, dfs_ohlcv['df_low'], target_future_days)
        df_data = pd.concat([df_data, future_min_var_before_max], axis='columns')
    
    return df_data

def add_output_loss_min_var(df, **hyperparams):
    sell_at_target = hyperparams.get('sell_at_target')
    
    if sell_at_target:
        df['output_loss_min_var'] = df['output_future_min_var_before_max']
    else:
        df['output_loss_min_var'] = df['output_future_min_var']

    return df

def add_output_is_loss_limit_reached(df, **hyperparams):
    loss_limit = hyperparams.get('loss_limit')

    df['output_is_loss_limit_reached'] = (df['output_loss_min_var'] <= loss_limit)

    return df

def add_output_var_class(df_data, **hyperparams):
    sell_at_target = hyperparams.get('sell_at_target')
    thresholds = hyperparams.get('thresholds')
    last_class = len(thresholds)

    if sell_at_target:
        output_class = classify_var(df_data[['output_future_max_var']], thresholds, 'output_var_class')
    else:
        output_class = classify_var(df_data[['output_future_end_var']], thresholds, 'output_var_class')

    output_class.loc[df_data['output_is_loss_limit_reached'], 'output_var_class'] = last_class
    
    df_data = pd.concat([df_data, output_class], axis='columns')

    return df_data

def add_output_is_buy(df, output_class_name, **hyperparams):
    accepted_n_first_classes = hyperparams.get('n_first_classes')[1]
    df['output_is_buy'] = (df[output_class_name] <= accepted_n_first_classes)
    
    return df

def add_output_profit(df, fee, **hyperparams):
    thresholds = hyperparams.get('thresholds')
    accepted_n_first_classes = hyperparams.get('n_first_classes')[1]
    loss_limit = hyperparams.get('loss_limit')
    sell_at_target = hyperparams.get('sell_at_target')

    accepted_var = thresholds[accepted_n_first_classes]

    loss_condition = df['output_loss_min_var'] <= loss_limit
    reached_target_condition = sell_at_target & (df['output_future_max_var'] > accepted_var)

    df['output_profit'] = np.select(
        [
            loss_condition,  # Condition for buy and loss condition
            reached_target_condition  # Condition for buy and reached target condition
        ],
        [
            loss_limit,  # Value if buy and meets loss condition
            accepted_var  # Value if buy and meets target condition
        ],
        default=df['output_future_end_var']  # Default value for buy condition not meeting the above
    )

    fee_coef = hf.get_fee_coef(fee)
    df['output_profit'] *= fee_coef
    
    return df

def add_future_rank(df_data, df_buy, **hyperparams):
    target_future_days = hyperparams.get('target_future_days')
    df_data['output_future_end_rank'] = inputs.get_rank(df_buy, past_days=0, future_days=target_future_days)
    
    return df_data

def add_output_rank_class(df_data, num_tickers, **hyperparams):
    rank_pct_thresholds = hyperparams.get('rank_pct_thresholds')
    rank_thresholds = np.floor(np.array(rank_pct_thresholds) * num_tickers).astype(int)
    
    output_class = classify_rank(df_data[['output_future_end_rank']], rank_thresholds, 'output_rank_class')
    
    df_data = pd.concat([df_data, output_class], axis='columns')

    return df_data

def add_outputs(df_data, df_buy, df_sell, dfs_ohlcv, num_tickers, output_class_name, fee, **hyperparams):
    df_data = add_future_vars(df_data, df_buy, df_sell, dfs_ohlcv, **hyperparams)
    df_data = add_output_loss_min_var(df_data, **hyperparams)
    df_data = add_output_is_loss_limit_reached(df_data, **hyperparams)
    df_data = add_output_var_class(df_data, **hyperparams)

    df_data = add_future_rank(df_data, df_buy, **hyperparams)
    df_data = add_output_rank_class(df_data, num_tickers, **hyperparams)

    df_data = add_output_is_buy(df_data, output_class_name, **hyperparams)
    df_data = add_output_profit(df_data, fee, **hyperparams)

    return df_data