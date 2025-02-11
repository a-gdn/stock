import config as cfg
import utils.helper_functions as hf

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Input, LSTM, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import AdamW # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import shap
import numpy as np
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'  # Disable file validation in the debugger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logs

def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop, errors='ignore')

def get_dfs_input_output(df_data, output_class_name):
    input_columns = [col for col in df_data.columns if col.startswith('input_')]
    df_input = df_data[input_columns]
    df_output = df_data[[output_class_name]]
    df_input = remove_highly_correlated_features(df_input)
    return df_input, df_output

def get_feature_importance(df_input, df_output):
    model = tf.keras.Sequential([
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(df_input, df_output, epochs=5, verbose=0)
    explainer = shap.Explainer(model, df_input)
    shap_values = explainer(df_input)
    importance_df = pd.DataFrame({'feature': df_input.columns, 'importance': np.abs(shap_values.values).mean(axis=0)})
    
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    logging.info(f"Top feature importances: {importance_df.head(10)}")
    return importance_df

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
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

def create_tf_model(**kwargs):
    logging.info("Creating TensorFlow model...")
    X_train = kwargs.get('X_train')
    X_test = kwargs.get('X_test')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')
    
    size_layer_1 = kwargs.get('size_layer_1', cfg.size_layer_1)
    size_layer_2 = kwargs.get('size_layer_2', cfg.size_layer_2)
    dropout_rate = kwargs.get('dropout_rate', cfg.dropout_rate)
    batch_size = kwargs.get('batch_size', cfg.batch_size)
    model_type = kwargs.get('model_type', cfg.model_type)

    if model_type == 'dense':
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(size_layer_1), BatchNormalization(), Activation('relu'), Dropout(dropout_rate),
            Dense(size_layer_2), BatchNormalization(), Activation('relu'), Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'lstm':
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(size_layer_1, return_sequences=True),
            LSTM(size_layer_2),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'transformer':
        inputs = Input(shape=(X_train.shape[1], 1))
        x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(size_layer_1, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(size_layer_2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.compile(optimizer=AdamW(learning_rate=cfg.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.early_stopping_patience, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=cfg.lr_reduction_factor, patience=cfg.lr_reduction_patience, min_lr=cfg.min_learning_rate)
    callbacks = [early_stopping, lr_scheduler]
    
    history = model.fit(X_train, y_train, epochs=cfg.max_epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)
    
    logging.info(f"Number of epochs used: {len(history.epoch)}")
    model.save(cfg.model_path)

def load_tf_model(df_data, hyperparams):
    logging.info("Loading TensorFlow model...")
    df_input, df_output = get_dfs_input_output(df_data, cfg.output_binary_name)
    get_feature_importance(df_input, df_output)
    test_train_data = get_test_train_data(df_input, df_output, cfg.test_size)
    
    if not os.path.exists(cfg.model_path) or not cfg.use_saved_model:
        tf.keras.backend.clear_session()
        create_tf_model(**{**test_train_data, **hyperparams})
    
    return test_train_data, tf.keras.models.load_model(cfg.model_path)
