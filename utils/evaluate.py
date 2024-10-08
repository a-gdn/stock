import config as cfg
import utils.helper_functions as hf

import numpy as np

def slice_df_test(df_data, test_size):
    return df_data.tail(test_size)

def add_predictions(df, model, X_test, **hyperparams):
    print(f'X_test shape: {X_test.shape}')
    
    predicted_n_first_classes = hyperparams['n_first_classes'][0]
    cumulated_probs_target = hyperparams['cumulated_probs_target']

    prediction_y_test_lists = model.predict(X_test)
    prediction_y_test_array = np.array(prediction_y_test_lists)
    df['prediction_probs'] = prediction_y_test_array.tolist()

    df['prediction_cumulated_probs'] = [sum(row[:predicted_n_first_classes+1]) for row in df['prediction_probs']]
    df['prediction_is_buy'] = (df['prediction_cumulated_probs'] > cumulated_probs_target)
    df['prediction_is_buy_is_correct'] = (df['output_is_buy'] == df['prediction_is_buy'])

    return df

def get_class_cumulative_percentages(y_test):
    unique_values, counts = np.unique(y_test, return_counts=True)
    percentages = counts / len(y_test)
    percentages = percentages[np.argsort(unique_values)]
    cumulative_percentages = np.cumsum(percentages)

    print(f'market cumulative % per class: {cumulative_percentages}')

    return cumulative_percentages

def get_market_rate(y_test, **hyperparams):
    accepted_n_first_classes = hyperparams['n_first_classes'][1]

    class_cumulative_percentages = get_class_cumulative_percentages(y_test)
    market_rate = class_cumulative_percentages[accepted_n_first_classes]

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

def get_performance_score(trimmed_average_profit, is_buy_count, num_tickers):
    estimated_days = cfg.test_size / num_tickers
    adjusted_profit = trimmed_average_profit ** 8 # to decrease small values, e.g. 0.8^2 = 0.64
    performance_score = adjusted_profit * min(is_buy_count, estimated_days)
    
    return performance_score

def evaluate_model(df_data, model, test_train_data, num_tickers, num_combinations, hyperparams):
    df_test = slice_df_test(df_data, cfg.test_size)
    df_test = add_predictions(df_test, model, test_train_data['X_test'], **hyperparams)
    
    market_rate = get_market_rate(test_train_data['y_test'], **hyperparams)

    binary_classification = get_binary_classification(df_test)
    
    df_prediction_is_buy = df_test[(df_test['prediction_is_buy'] == True)]
    if (not cfg.use_hyperopt and num_combinations == 1):
        print(df_prediction_is_buy.to_markdown())
        df_prediction_is_buy.to_excel(f'./outputs/{hf.get_date()}_classifier_df_prediction_is_buy.xlsx')

    profits = get_profits(df_prediction_is_buy)
    prediction_is_buy_count = len(df_prediction_is_buy['output_profit'])
    loss_limit_reached_pct = get_loss_limit_pct(df_prediction_is_buy)
    performance_score = get_performance_score(profits['trimmed_average_profit'],
                                              prediction_is_buy_count, num_tickers)

    performance_metrics = {
        'performance_score': performance_score,
        **profits,
        'prediction_is_buy_count': prediction_is_buy_count,
        'loss_limit_reached_pct': loss_limit_reached_pct,
        'market_rate': market_rate,
        **binary_classification,
        'winning_rate_vs_market': binary_classification['winning_rate'] - market_rate,
    }

    return performance_metrics