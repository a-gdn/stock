import hyperopt as hp
from hyperopt import hp, fmin, tpe

db_path = './db/ohlcv_ntickers_593_2000-08-01_to_2024-06-15.pkl'
transformed_data_path = './outputs/classifier_transformed_data.pkl'
model_path = './outputs/classifier_model.keras'

fee = 0.002

use_hyperopt = True
use_saved_transformed_data = False
use_saved_model = False

start_date = '2008-01-01' #'2013-01-01'
test_size = 60000
epochs = 2
hyperopt_n_iterations = 400

output_class_name = 'output_var_class' #'output_var_class' or 'output_rank_class'

param_grid = {
    'buying_time': ['Open'], 'selling_time': ['Close'], #'Open', 
    'target_future_days': [0], 'loss_limit': [0.997],
    'sell_at_target': [True],
    'size_layer_1': [128], 'size_layer_2': [128], 'size_layer_3': [128],
    'dropout_rate': [0.1], 'balance_data': [True], 'batch_size': [32], #'dropout_rates': [i for i in list(np.arange(0, 0.3, 0.1))], 'batch_sizes': [32, 64, 128],
    'n_first_classes': [[0,0]],
    'cumulated_probs_target': [0.7,0.8,0.9],
    'thresholds': [[1.02],[1.03],[1.04]], #,[1.01],[1.04]
    'rank_pct_thresholds': [[0.45]]
}

search_space = {
    'buying_time': hp.choice('buying_time', ['Close']),
    'selling_time': hp.choice('selling_time', ['Open', 'Close']),
    'target_future_days': hp.randint('target_future_days', 1, 30), #hp.randint('target_future_days', 1, 60), #1, 60
    'loss_limit': hp.uniform('loss_limit', 0.8, 1),
    'sell_at_target': hp.choice('sell_at_target', [True, False]), #[True, False]
    'size_layer_1': hp.choice('size_layer_1', [128]),
    'size_layer_2': hp.choice('size_layer_2', [128]),
    'size_layer_3': hp.choice('size_layer_3', [128]), #[64, 128, 256]
    'dropout_rate': hp.uniform('dropout_rate', 0.02, 0.25), #hp.uniform('dropout_rate', 0.05, 0.1), #0, 0.3
    'balance_data': hp.choice('balance_data', [True, False]),
    'batch_size': hp.choice('batch_size', [128]), #[32, 64, 128]
    'n_first_classes': hp.choice('n_first_classes', [[0, 0]]),
    'cumulated_probs_target': hp.uniform('cumulated_probs_target', 0.6, 1),
    'thresholds': hp.choice('thresholds', [[1], [1.005], [1.01], [1.015], [1.02], [1.025], [1.03], [1.035], [1.04], [1.05], [1.06], [1.08], [1.10], [1.12], [1.15]]),
    # 'thresholds': hp.choice('thresholds', [[1.08, 1.04, 1.02, 1], [1.06, 1.03, 1.01], [1.05, 1.025, 1], [1.1, 1.05, 1.01]]),
    # 'rank_pct_thresholds': hp.uniform('rank_pct_thresholds', 0.002, 0.5),
    'rank_pct_thresholds': hp.choice('rank_pct_thresholds', [[0.08, 0.2, 0.33]])
}
