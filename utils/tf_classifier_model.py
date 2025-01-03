import config as cfg
import utils.helper_functions as hf

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import Counter

import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1' # disable file validation in the debugger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #0: All logs (default setting), 1: Filter out INFO logs, up to 3

def get_dfs_input_output(df_data, output_class_name):
    input_columns = [col for col in df_data.columns if col.startswith('input_')]
    df_input = df_data[input_columns]
    df_output = df_data[[output_class_name]]

    return df_input, df_output

def get_test_train_data(df_input, df_output, test_size):
    X_train = df_input[:-test_size].values
    y_train = df_output[:-test_size].values.ravel().astype(int)

    X_test = df_input.tail(test_size).values
    y_test = df_output.tail(test_size).values.ravel().astype(int)

    if cfg.use_hyperopt and X_train.shape[0] == 0:
        raise ValueError("Empty training set, skipping this trial")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hf.save_object(scaler, './outputs/scaler.pkl')

    # print(f"number of elements in y_train: {len(y_train)}")
    # print(f"number of elements in y_test: {len(y_test)}")

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

def create_tf_model(**kwargs):
    X_train = kwargs.get('X_train')
    X_test = kwargs.get('X_test')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')
    
    size_layer_1 = kwargs.get('size_layer_1')
    size_layer_2 = kwargs.get('size_layer_2')
    size_layer_3 = kwargs.get('size_layer_3')
    dropout_rate = kwargs.get('dropout_rate')
    balance_data = kwargs.get('balance_data')
    batch_size = kwargs.get('batch_size')

    # last_layers_size = len(thresholds) + 1

    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))

    # model.add(Dense(size_layer_1, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(size_layer_1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(size_layer_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(size_layer_3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    # model.add(Dense(last_layers_size, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if (balance_data):
        counter = Counter(y_train)
        max_count = max(counter.values())
        class_weights = {cls: max_count / count for cls, count in counter.items()}
        model.fit(X_train, y_train, epochs=cfg.epochs, batch_size=batch_size, validation_data=(X_test, y_test), class_weight=class_weights)
    else:
        model.fit(X_train, y_train, epochs=cfg.epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    model.save(cfg.model_path)

def load_tf_model(df_data, hyperparams):
    df_input, df_output = get_dfs_input_output(df_data, cfg.output_binary_name)

    test_train_data = get_test_train_data(df_input, df_output, cfg.test_size)

    if not os.path.exists(cfg.model_path) or not cfg.use_saved_model:
        # print(f'need to create {cfg.model_path}')
        tf.keras.backend.clear_session()
        create_tf_model(**{**test_train_data, **hyperparams})
    
    tf_model = tf.keras.models.load_model(cfg.model_path)

    return test_train_data, tf_model