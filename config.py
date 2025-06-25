import hyperopt as hp
from hyperopt import hp, fmin, tpe

db_path = './db/merged_ohlcv_fundamentals.pkl'
sp500_db_path = './db/sp500_2000-08-01_to_2024-11-20.pkl'
vix_db_path = './db/vix_2000-08-01_to_2024-11-20.pkl'

transformed_data_path = './outputs/classifier_transformed_data.pkl'
model_path = './outputs/classifier_model.keras'
trials_path = './outputs/trials.pkl'
results_path = './outputs/results.xlsx'

fee = 0.002

use_hyperopt = False
use_saved_transformed_data = False
use_saved_model = False

start_date = '2015-06-01' #'2013-01-01'
test_size = 60000

max_epochs = 3
early_stopping_patience = 3
lr_reduction_factor = 0.1
lr_reduction_patience = 2
min_learning_rate = 1e-6
learning_rate = 1e-3

hyperopt_n_iterations = 1000
save_every_n_iterations = 10

output_binary_name = 'output_var_binary' #'output_var_binary' or 'output_rank_binary'

param_grid = {
    'buying_time': ['Close'], 'selling_time': ['Close'], #'Open', 
    'target_future_days': [44],
    'loss_limit': [0.2], #0.4, 0.55, 0.7, 
    'sell_at_target': [False],
    'size_layer_1': [128], 'size_layer_2': [64], 'size_layer_3': [64],
    'dropout_rate': [0.35], 'use_focal_loss': [True], 'batch_size': [128], #'dropout_rates': [i for i in list(np.arange(0, 0.3, 0.1))], 'batch_sizes': [32, 64, 128],
    'confidence_threshold': [0.7],
    'var_threshold': [1.005],
    'rank_pct_threshold': [0.45]
}

search_space = {
    'buying_time': hp.choice('buying_time', ['Open', 'Close']),
    'selling_time': hp.choice('selling_time', ['Open', 'Close']),
    'target_future_days': hp.randint('target_future_days', 0, 60), #hp.randint('target_future_days', 1, 60), #1, 60
    'loss_limit': hp.uniform('loss_limit', 0, 1),
    'sell_at_target': hp.choice('sell_at_target', [False, False]), #[True, False] or [False, False]
    'size_layer_1': hp.choice('size_layer_1', [128]),
    'size_layer_2': hp.choice('size_layer_2', [64]),
    'size_layer_3': hp.choice('size_layer_3', [64]), #[64, 128, 256]
    'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.5), #0, 0.3
    'use_focal_loss': hp.choice('use_focal_loss', [True, True]),
    'batch_size': hp.choice('batch_size', [128]), #[32, 64, 128]
    'confidence_threshold': hp.uniform('confidence_threshold', 0.5, 1),
    'var_threshold': hp.uniform('var_threshold', 1, 3),
    'rank_pct_threshold': hp.uniform('rank_pct_threshold', 0, 1)
}
