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

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'  # Disable file validation in the debugger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: All logs (default setting), 1: Filter out INFO logs, up to 3
pd.options.mode.copy_on_write = True  # Avoid making unnecessary copies of DataFrames or Series

num_combinations = cfg.hyperopt_n_iterations if cfg.use_hyperopt else hf.get_num_combinations(cfg.param_grid)

print(num_combinations)

def is_valid_combination(hyperparams):
    return hyperparams['target_future_days'] != 0 or (hyperparams['buying_time'] == 'Open' and hyperparams['selling_time'] == 'Close')

df = pd.read_pickle(cfg.db_path)
df = hf.get_rows_after_date(df, cfg.start_date)
df = hf.fillnavalues(df)

def get_single_level_df(df, ohlcv):
    new_df = df[[ohlcv]]
    new_df = hf.remove_top_column_name(new_df)
    return new_df

def get_ohlcv_dfs(df):
    df_open = get_single_level_df(df, 'Open')
    df_high = get_single_level_df(df, 'High')
    df_low = get_single_level_df(df, 'Low')
    df_close = get_single_level_df(df, 'Close')
    df_volume = get_single_level_df(df, 'Volume')
    
    return {'df_open': df_open, 'df_high': df_high, 'df_low': df_low,
            'df_close': df_close, 'df_volume': df_volume}

num_tickers = hf.get_num_tickers(get_single_level_df(df, 'Open'))
print(f'Number of tickers: {num_tickers}')

def get_df_data(hyperparams):
    df_buy = get_single_level_df(df, hyperparams['buying_time'])
    df_sell = get_single_level_df(df, hyperparams['selling_time'])
    dfs_ohlcv = get_ohlcv_dfs(df)

    if os.path.exists(cfg.transformed_data_path) and cfg.use_saved_transformed_data:
        df_data = pd.read_pickle(cfg.transformed_data_path)
    else:
        df_data = inputs.get_inputs(df_buy, dfs_ohlcv, hyperparams['buying_time'])
        df_data.to_pickle(cfg.transformed_data_path)

    df_data = outputs.add_outputs(df_data, df_buy, df_sell, dfs_ohlcv, num_tickers, cfg.output_binary_name, cfg.fee, **hyperparams)
    df_data = df_data.dropna()

    return df_data

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

from itertools import product

def get_model_result(hyperparams):
    print(f"Evaluating hyperparameters: {hyperparams}")

    df_data = get_df_data(hyperparams)
    test_train_data, model = tf_classifier_model.load_tf_model(df_data, hyperparams)
    performance_metrics = eval.evaluate_model(df_data, model, test_train_data, num_tickers, num_combinations, hyperparams)

    result = {**performance_metrics, **hyperparams}
    print(f"Result: {result}")

    release_memory(df_data, model, test_train_data)

    return result

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
    print_results(results)

caffeinate_process = subprocess.Popen(["caffeinate", "-dims"])
print("Caffeinate started...")

try:
    main()
finally:
    caffeinate_process.terminate()
    print("Caffeinate stopped.")

if __name__ == "__main__":
    pass
