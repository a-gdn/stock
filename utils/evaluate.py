import config as cfg
import utils.helper_functions as hf

import numpy as np

def slice_df_test(df_data, test_size):
    return df_data.tail(test_size)

def add_predictions(df, model, X_test, **hyperparams):
    print(f'X_test shape: {X_test.shape}')
    
    confidence_threshold = hyperparams['confidence_threshold']

    prediction_y_test_array = model.predict(X_test).flatten()
    df['prediction_probs'] = prediction_y_test_array.tolist()

    df['prediction_is_buy'] = (df['prediction_probs'] > confidence_threshold)

    if cfg.output_binary_name not in {'output_var_binary', 'output_rank_binary'}:
        raise ValueError(f'output_binary_name must be either "output_var_binary" or "output_rank_binary"')

    df['prediction_is_buy_is_correct'] = (df[cfg.output_binary_name] == df['prediction_is_buy'])

    return df

def get_market_rate(y_test):
    market_rate = np.sum(y_test) / len(y_test) # Calculate the proportion of positive class (1's) in y_test
    return market_rate

def get_binary_classification(df):
    # tp: true positive, tn: true negative, fp: false positive, fn: false negative  
    tp = ((df['output_is_buy'] == True) & (df['prediction_is_buy'] == True)).sum()
    tn = ((df['output_is_buy'] == False) & (df['prediction_is_buy'] == False)).sum()
    fp = ((df['output_is_buy'] == False) & (df['prediction_is_buy'] == True)).sum()
    fn = ((df['output_is_buy'] == True) & (df['prediction_is_buy'] == False)).sum()

    winning_rate = float(tp / (tp + fp)) if (tp + fp) > 0 else 0

    return {
        'true_positives': tp, 'true_negatives': tn,
        'false_positives': fp, 'false_negatives': fn,
        'winning_rate': winning_rate
    }

def get_profits(df_prediction_is_buy):
    trimmed_average_profit = hf.get_trimmed_average(df_prediction_is_buy['output_profit'], pct_to_trim=0.03, min_num_to_trim=8)
    average_profit = df_prediction_is_buy['output_profit'].mean()
    median_profit = df_prediction_is_buy['output_profit'].median()

    return {
        'trimmed_average_profit': trimmed_average_profit,
        'average_profit': average_profit,
        'median_profit': median_profit
    }

def get_loss_limit_pct(df):
    return df['output_is_loss_limit_reached'].sum() / len(df) if len(df) > 0 else 0

def get_profitable_rate(df):
    profitable_count = (df['output_profit'] > 1).sum()
    total_count = df['output_profit'].count()
    profitable_rate = round((profitable_count / total_count) * 100, 2) if total_count > 0 else 0

    return profitable_rate

def get_performance_score(trimmed_average_profit, profitable_rate, is_buy_count, num_tickers, **hyperparams):
    # estimated_total_days = cfg.test_size / num_tickers
    # holding_total_days = min(is_buy_count, estimated_total_days)
    # holding_rate = holding_total_days / estimated_total_days

    min_is_buy_count = 125
    is_buy_count_score = min(1, is_buy_count / min_is_buy_count)

    # adjusted_profit = trimmed_average_profit # to decrease small values, e.g. 0.8 ** 2 = 0.8^2 = 0.64
    # performance_score = trimmed_average_profit ** (investment_total_days / max(1, stock_holding_days))
    performance_score = is_buy_count_score * profitable_rate * trimmed_average_profit**2

    # if trimmed_average_profit < 1:
    #     performance_score /= 100
    
    return performance_score

def evaluate_model(df_data, model, test_train_data, num_tickers, num_combinations, hyperparams):
    df_test = slice_df_test(df_data, cfg.test_size)
    df_test = add_predictions(df_test, model, test_train_data['X_test'], **hyperparams)
    
    market_rate = get_market_rate(test_train_data['y_test'])
    binary_classification = get_binary_classification(df_test)

    if df_test['prediction_is_buy'].any():
        df_prediction_is_buy = df_test[(df_test['prediction_is_buy'] == True)]
        if (not cfg.use_hyperopt and num_combinations == 1):
            print(df_prediction_is_buy.to_markdown())
            df_prediction_is_buy.to_excel(f'./outputs/{hf.get_date()}_classifier_df_prediction_is_buy.xlsx')

        profits = get_profits(df_prediction_is_buy)
        prediction_is_buy_count = len(df_prediction_is_buy['output_profit'])
        loss_limit_reached_pct = get_loss_limit_pct(df_prediction_is_buy)
        profitable_rate = get_profitable_rate(df_prediction_is_buy)
        performance_score = get_performance_score(profits['trimmed_average_profit'], profitable_rate, prediction_is_buy_count,
                                                  num_tickers, **hyperparams)

        performance_metrics = {
            'performance_score': performance_score,
            **profits,
            'prediction_is_buy_count': prediction_is_buy_count,
            'loss_limit_reached_pct': loss_limit_reached_pct,
            **binary_classification,
            'market_rate': market_rate,
            'winning_rate_vs_market': binary_classification['winning_rate'] - market_rate,
            'profitable_rate': profitable_rate
        }
    else:
        performance_metrics = {
            'performance_score': 0,
            'trimmed_average_profit': 1,
            'average_profit': 1,
            'median_profit': 1,
            'prediction_is_buy_count': 0,
            'loss_limit_reached_pct': 0,
            **binary_classification,
            'market_rate': market_rate,
            'winning_rate_vs_market': binary_classification['winning_rate'] - market_rate,
            'profitable_rate': 0
        }

    return performance_metrics