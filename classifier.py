import config as cfg
from utils import helper_functions as hf
from utils import inputs
from utils import outputs
from utils import tf_classifier_model
from utils import evaluate as eval

from IPython.display import display, clear_output

import pandas as pd
import numpy as np

import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_FAIL, STATUS_OK, Trials

import os
import subprocess 
import gc

from itertools import product

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'  # Disable file validation in the debugger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0: All logs (default setting), 1: Filter out INFO logs, up to 3
pd.options.mode.copy_on_write = True  # Avoid making unnecessary copies of DataFrames or Series

num_combinations = cfg.hyperopt_n_iterations if cfg.use_hyperopt else hf.get_num_combinations(cfg.param_grid)

print(num_combinations)

def is_valid_combination(hyperparams):
    return hyperparams['target_future_days'] != 0 or (hyperparams['buying_time'] == 'Open' and hyperparams['selling_time'] == 'Close')

def load_db(path):    
    df = pd.read_pickle(path)
    df = hf.get_rows_after_date(df, cfg.start_date)
    df = hf.fillnavalues(df)
    return df

def get_single_level_df(df, ohlcv):
    new_df = df[[ohlcv]]
    new_df = hf.remove_top_column_name(new_df)
    return new_df

def get_ohlcv_dfs(df, source, **hyperparams):
    df_buy = get_single_level_df(df, hyperparams["buying_time"])
    df_sell = get_single_level_df(df, hyperparams["selling_time"])

    df_open = get_single_level_df(df, 'Open')
    df_high = get_single_level_df(df, 'High')
    df_low = get_single_level_df(df, 'Low')
    df_close = get_single_level_df(df, 'Close')
    df_volume = get_single_level_df(df, 'Volume')
    
    return {
        f'df_{source}_buy': df_buy, f'df_{source}_sell': df_sell,
        f'df_{source}_open': df_open, f'df_{source}_high': df_high, f'df_{source}_low': df_low, f'df_{source}_close': df_close, f'df_{source}_volume': df_volume,
    }

def get_fundamental_dfs(df):
    df_current_ratio = get_single_level_df(df, 'current_ratio')
    df_ev_to_ebitda_ltm = get_single_level_df(df, 'ev_to_ebitda_ltm')
    df_fcf_yield_ltm = get_single_level_df(df, 'fcf_yield_ltm')
    df_marketcap = get_single_level_df(df, 'marketcap')
    df_pe_ltm = get_single_level_df(df, 'pe_ltm')
    df_price_to_book = get_single_level_df(df, 'price_to_book')
    df_roa = get_single_level_df(df, 'roa')
    df_roe = get_single_level_df(df, 'roe')
    df_total_debt = get_single_level_df(df, 'total_debt')
    df_total_rev = get_single_level_df(df, 'total_rev')

    df_current_ratio_var_1 = get_single_level_df(df, 'current_ratio_var_1')
    df_ev_to_ebitda_ltm_var_1 = get_single_level_df(df, 'ev_to_ebitda_ltm_var_1')
    df_fcf_yield_ltm_var_1 = get_single_level_df(df, 'fcf_yield_ltm_var_1')
    df_marketcap_var_1 = get_single_level_df(df, 'marketcap_var_1')
    df_pe_ltm_var_1 = get_single_level_df(df, 'pe_ltm_var_1')
    df_price_to_book_var_1 = get_single_level_df(df, 'price_to_book_var_1')
    df_roa_var_1 = get_single_level_df(df, 'roa_var_1')
    df_roe_var_1 = get_single_level_df(df, 'roe_var_1')
    df_total_debt_var_1 = get_single_level_df(df, 'total_debt_var_1')
    df_total_rev_var_1 = get_single_level_df(df, 'total_rev_var_1')

    df_current_ratio_var_2 = get_single_level_df(df, 'current_ratio_var_2')
    df_ev_to_ebitda_ltm_var_2 = get_single_level_df(df, 'ev_to_ebitda_ltm_var_2')
    df_fcf_yield_ltm_var_2 = get_single_level_df(df, 'fcf_yield_ltm_var_2')
    df_marketcap_var_2 = get_single_level_df(df, 'marketcap_var_2')
    df_pe_ltm_var_2 = get_single_level_df(df, 'pe_ltm_var_2')
    df_price_to_book_var_2 = get_single_level_df(df, 'price_to_book_var_2')
    df_roa_var_2 = get_single_level_df(df, 'roa_var_2')
    df_roe_var_2 = get_single_level_df(df, 'roe_var_2')
    df_total_debt_var_2 = get_single_level_df(df, 'total_debt_var_2')
    df_total_rev_var_2 = get_single_level_df(df, 'total_rev_var_2')

    df_current_ratio_var_4 = get_single_level_df(df, 'current_ratio_var_4')
    df_ev_to_ebitda_ltm_var_4 = get_single_level_df(df, 'ev_to_ebitda_ltm_var_4')
    df_fcf_yield_ltm_var_4 = get_single_level_df(df, 'fcf_yield_ltm_var_4')
    df_marketcap_var_4 = get_single_level_df(df, 'marketcap_var_4')
    df_pe_ltm_var_4 = get_single_level_df(df, 'pe_ltm_var_4')
    df_price_to_book_var_4 = get_single_level_df(df, 'price_to_book_var_4')
    df_roa_var_4 = get_single_level_df(df, 'roa_var_4')
    df_roe_var_4 = get_single_level_df(df, 'roe_var_4')
    df_total_debt_var_4 = get_single_level_df(df, 'total_debt_var_4')
    df_total_rev_var_4 = get_single_level_df(df, 'total_rev_var_4')

    return {
        'df_current_ratio': df_current_ratio, 'df_ev_to_ebitda_ltm': df_ev_to_ebitda_ltm, 'df_fcf_yield_ltm': df_fcf_yield_ltm, 'df_marketcap': df_marketcap, 'df_pe_ltm': df_pe_ltm, 'df_price_to_book': df_price_to_book, 'df_roa': df_roa, 'df_roe': df_roe, 'df_total_debt': df_total_debt, 'df_total_rev': df_total_rev,
        'df_current_ratio_var_1': df_current_ratio_var_1, 'df_ev_to_ebitda_ltm_var_1': df_ev_to_ebitda_ltm_var_1, 'df_fcf_yield_ltm_var_1': df_fcf_yield_ltm_var_1, 'df_marketcap_var_1': df_marketcap_var_1, 'df_pe_ltm_var_1': df_pe_ltm_var_1, 'df_price_to_book_var_1': df_price_to_book_var_1, 'df_roa_var_1': df_roa_var_1, 'df_roe_var_1': df_roe_var_1, 'df_total_debt_var_1': df_total_debt_var_1, 'df_total_rev_var_1': df_total_rev_var_1,
        'df_current_ratio_var_2': df_current_ratio_var_2, 'df_ev_to_ebitda_ltm_var_2': df_ev_to_ebitda_ltm_var_2, 'df_fcf_yield_ltm_var_2': df_fcf_yield_ltm_var_2, 'df_marketcap_var_2': df_marketcap_var_2, 'df_pe_ltm_var_2': df_pe_ltm_var_2, 'df_price_to_book_var_2': df_price_to_book_var_2, 'df_roa_var_2': df_roa_var_2, 'df_roe_var_2': df_roe_var_2, 'df_total_debt_var_2': df_total_debt_var_2, 'df_total_rev_var_2': df_total_rev_var_2,
        'df_current_ratio_var_4': df_current_ratio_var_4, 'df_ev_to_ebitda_ltm_var_4': df_ev_to_ebitda_ltm_var_4, 'df_fcf_yield_ltm_var_4': df_fcf_yield_ltm_var_4, 'df_marketcap_var_4': df_marketcap_var_4, 'df_pe_ltm_var_4': df_pe_ltm_var_4, 'df_price_to_book_var_4': df_price_to_book_var_4, 'df_roa_var_4': df_roa_var_4, 'df_roe_var_4': df_roe_var_4, 'df_total_debt_var_4': df_total_debt_var_4, 'df_total_rev_var_4': df_total_rev_var_4
    }

def get_df_data(hyperparams):
    stock_df = load_db(cfg.db_path)
    market_df = load_db(cfg.market_db_path)

    market_df = market_df.reindex(stock_df.index)
    market_df.ffill(inplace=True)
    market_df.bfill(inplace=True)

    num_stocks = hf.get_num_tickers(get_single_level_df(stock_df, 'Open'))
    print(f'Number of stocks: {num_stocks}')
    
    stock_ohlcv_dfs = get_ohlcv_dfs(stock_df, "stock", **hyperparams)
    stock_fundamental_dfs = get_fundamental_dfs(stock_df)
    market_ohlcv_dfs = get_ohlcv_dfs(market_df, "market", **hyperparams)
    dfs = {**stock_ohlcv_dfs, **stock_fundamental_dfs, **market_ohlcv_dfs}

    if os.path.exists(cfg.transformed_data_path) and cfg.use_saved_transformed_data:
        df_data = pd.read_pickle(cfg.transformed_data_path)
    else:
        df_data = inputs.get_inputs(dfs, hyperparams['buying_time'])
        df_data.to_pickle(cfg.transformed_data_path)

    df_data = outputs.add_outputs(df_data, dfs, num_stocks, **hyperparams)    

    df_data = hf.fillnavalues(df_data)
    print(f"df_data shape: {df_data.shape}")

    return df_data, num_stocks

def load_results(path):
    if os.path.exists(path):
        print(f"Loading results from {path}")
        return pd.read_excel(path).to_dict(orient='records')
    return []

def load_trials(path):
    if os.path.exists(path):
        print(f"Loading trials from {path}")
        return pd.read_pickle(path)
    return Trials()

def save_results(results, trials, results_path, trials_path):
    pd.DataFrame(results).to_excel(results_path, index=False)  

    if trials is not None:
        pd.to_pickle(trials, trials_path)

def save_results_if_needed(results, trials, iteration, total_iterations, results_path, trials_path):
    if iteration % cfg.save_every_n_iterations == 0 or iteration == total_iterations:
        save_results(results, trials, results_path, trials_path)
        clear_output(wait=True)

def print_results(results):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df_results = pd.DataFrame(results)
    if 'performance_score' in df_results.columns:
        df_results = df_results.sort_values(by='performance_score', ascending=False)
    else:
        print("Warning: 'performance_score' column is not yet in the results. You may need to delete trials.pkl and results.xlsx to start fresh.")
    print(df_results.head(1000))

def release_memory(*args):
    for var in args:
        del var
    gc.collect()

def get_model_result(hyperparams):
    print(f"Evaluating hyperparameters: {hyperparams}")

    df_data, num_tickers = get_df_data(hyperparams)

    all_fold_performance_metrics = [] # Initialize list to store results from each fold

    total_rows = len(df_data)

    # Test the model with a sliding window approach
    for i, train_end_index in enumerate(range(cfg.train_window_size, total_rows, cfg.test_window_size)):
        if (train_end_index + cfg.test_window_size) > total_rows:
            print(f"Not enough data for a full test window starting at index {train_end_index}. Breaking.")
            break

        print(f"\n--- Fold {i+1}: Training on rows 1 to {train_end_index}, Testing on rows {train_end_index + 1} to {train_end_index + cfg.test_window_size} ---")

        test_train_data, model = tf_classifier_model.load_tf_model(df_data, train_end_index, hyperparams)

        performance_metrics = eval.evaluate_model(df_data, model, test_train_data, num_tickers, num_combinations, hyperparams)

        all_fold_performance_metrics.append(performance_metrics)

        print(f"Result for Fold {i+1}:")
        for k, v in performance_metrics.items():
            print(f"- {k}: {v}")

        release_memory(model, test_train_data) # Release memory for each fold

    # After all folds are processed, aggregate (average) the results
    df_fold_metrics = pd.DataFrame(all_fold_performance_metrics)    
    aggregated_results = df_fold_metrics.mean().round(2).to_dict()

    filtered_df = df_fold_metrics[df_fold_metrics['prediction_is_buy_count'] > 0]

    if not filtered_df.empty:
        performance_score_mean = filtered_df['performance_score'].mean().round(2)
    else:
        performance_score_mean = 0
    
    aggregated_results['performance_score'] = performance_score_mean
    
    print(f"\nAggregated Results (across {len(all_fold_performance_metrics)} folds):")
    for k, v in aggregated_results.items():
        print(f"- {k}: {v}")
    
    final_result = {**aggregated_results, **hyperparams}

    release_memory(df_data) # Release memory for df_data after all folds

    return final_result

def hyperopt_search(results):
    def objective(hyperparams):   
        try:
            if not is_valid_combination(hyperparams):
                print("Invalid hyperparameter combination.")
                return {'loss': float('inf'), 'status': STATUS_FAIL}
            
            print(f"Trial {len(trials)}/{cfg.hyperopt_n_iterations}")
            result = get_model_result(hyperparams)
            results.append(result)
            save_results_if_needed(results, trials, len(trials), cfg.hyperopt_n_iterations, cfg.results_path, cfg.trials_path)

            performance = result.get('performance_score', None)
            if performance is None:
                print("Missing performance score in result.")
                return {'loss': float('inf'), 'status': STATUS_FAIL}

            return {'loss': -performance, 'status': STATUS_OK}
        
        except Exception as e:
            print(f'Skipping trial, error: {e}')
            return {'status': STATUS_FAIL}

    trials = load_trials(cfg.trials_path)
    best = fmin(
        fn=objective,
        space=cfg.search_space,
        algo=tpe.suggest,
        max_evals=cfg.hyperopt_n_iterations,
        trials=trials
    )
    print(f'Best parameters: {best}')
    return trials

def grid_search(results):
    param_combinations = list(product(*cfg.param_grid.values()))

    for i, params in enumerate(param_combinations, start=1):
        hf.print_combination(i, num_combinations)
        hyperparams = dict(zip(cfg.param_grid.keys(), params))

        try:
            if not is_valid_combination(hyperparams):
                print(f"Skipping invalid combination {i}/{num_combinations}: {hyperparams}")
                continue
            
            result = get_model_result(hyperparams)
            results.append(result)
            save_results_if_needed(results, None, i, num_combinations, cfg.results_path, cfg.trials_path)

        except Exception as e:
            print(f"Error at combination {i}/{num_combinations}: {e}")

def main():
    results = load_results(cfg.results_path)

    if cfg.use_hyperopt:
        trials = hyperopt_search(results)
    else:
        trials = None
        grid_search(results)

    save_results(results, trials, cfg.results_path, cfg.trials_path)
    # print_results(results)

with subprocess.Popen(["caffeinate", "-dims"]) as caffeinate_process:
    print("Caffeinate started...")
    try:
        main()
    finally:
        print("Caffeinate stopped.")

if __name__ == "__main__":
    pass
