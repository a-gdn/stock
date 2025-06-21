import config as cfg
import utils.helper_functions as hf

import tensorflow as tf
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Input # type: ignore
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

def remove_highly_correlated_features(df, importance_df, threshold=0.9):
    """
    Removes highly correlated features from a DataFrame, prioritizing dropping the
    least important features based on SHAP values.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        importance_df (pd.DataFrame): DataFrame with feature names and their SHAP importances.
                                       Must have columns 'feature' and 'importance'.
        threshold (float, optional): The correlation threshold.  Defaults to 0.9.

    Returns:
        pd.DataFrame: A DataFrame with highly correlated features removed.
    """

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    dropped_features = []  # Keep track of dropped features

    while to_drop:
        feature_to_evaluate = to_drop[0] # examine the first feature that can be dropped
        correlated_features = upper.index[upper[feature_to_evaluate] > threshold].tolist() # get other correlated features

        if len(correlated_features) > 1: # if there is a pair of correlated features
            # Find the least important feature among the correlated ones based on SHAP values
            feature_importances = importance_df[importance_df['feature'].isin(correlated_features)]
            least_important_feature = feature_importances.sort_values(by='importance', ascending=True)['feature'].iloc[0]

            # Drop the least important feature and update the upper correlation matrix and the list of features to drop
            df.drop(columns=[least_important_feature], inplace=True, errors='ignore')
            dropped_features.append(least_important_feature)

            #update the list of features that can be dropped
            to_drop.remove(feature_to_evaluate)
            if least_important_feature in to_drop:
                to_drop.remove(least_important_feature)

            # Recalculate the correlation matrix and upper triangle
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        else: # if there are no correlated features to the feature to drop
            df.drop(columns=[feature_to_evaluate], inplace=True, errors='ignore')
            dropped_features.append(feature_to_evaluate)

            to_drop.remove(feature_to_evaluate)

            # Recalculate the correlation matrix and upper triangle
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


    if dropped_features:
        logging.info(f"Dropped highly correlated features (prioritizing least important): {dropped_features}")
    else:
        logging.info("No highly correlated features found to drop.")

    return df

def get_dfs_input_output(df_data, output_class_name):
    input_columns = [col for col in df_data.columns if col.startswith('input_')]
    df_input = df_data[input_columns]
    df_output = df_data[[output_class_name]]
    return df_input, df_output

def get_feature_importance(df_input, df_output, sample_size=10000):
    # Subsample the data to reduce SHAP computation time
    if len(df_input) > sample_size:
        df_sample_input = df_input.sample(n=sample_size, random_state=42)
        df_sample_output = df_output.loc[df_sample_input.index]
    else:
        df_sample_input, df_sample_output = df_input, df_output

    # Define & Train the Model (the model should be simple)
    model = tf.keras.Sequential([
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(df_sample_input, df_sample_output, epochs=1, verbose=0)

    # Use SHAP PermutationExplainer
    explainer = shap.PermutationExplainer(model.predict, df_sample_input)
    shap_values = explainer(df_sample_input)

    # Compute Mean Absolute SHAP Values
    importance_df = pd.DataFrame({
        'feature': df_sample_input.columns,
        'importance': np.abs(shap_values.values).mean(axis=0)
    })
    
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    logging.info(f"Top feature importances: {importance_df.head(10)}")

    return importance_df

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for binary classification.
    Args:
        gamma (float): Focusing parameter to down-weight easy examples. Default is 2.0.
        alpha (float): Balance parameter to adjust for class imbalance. Default is 0.25.
    Returns:
        Loss function that can be used in model compilation.
    """

    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to avoid log(0) issues
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        
        # Cross entropy loss for each example
        cross_entropy_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Compute the modulating factor (1 - p_t)^gamma
        loss_modulation = tf.pow((1 - y_pred), gamma)
        
        # Compute the focal loss
        loss = alpha * loss_modulation * cross_entropy_loss
        
        return tf.reduce_mean(loss)

    return focal_loss_fixed

def get_test_train_data(df_input, df_output, test_size):
    """
    Splits data into training and testing sets, scales the input features,
    and ensures that the test size is valid.

    Args:
        df_input (pd.DataFrame): DataFrame containing the input features.
        df_output (pd.DataFrame): DataFrame containing the output variable.
        test_size (int): The number of samples to use for the test set.

    Returns:
        dict: A dictionary containing the training and testing data (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If the test_size is not a positive integer or is greater than or equal to the total number of samples.
    """

    n_samples = len(df_input)

    # Input validation
    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError("test_size must be a positive integer.")
    if test_size >= n_samples:
        raise ValueError(f"test_size ({test_size}) must be smaller than the total number of samples ({n_samples}).")
    
    X_train = df_input[:-test_size].values
    y_train = df_output[:-test_size].values.ravel().astype(int)
    X_test = df_input.tail(test_size).values
    y_test = df_output.tail(test_size).values.ravel().astype(int)

    if cfg.use_hyperopt and X_train.shape[0] == 0:
        raise ValueError("Empty training set, skipping this trial")
    
    print("NaN in X_train:", np.isnan(X_train).sum())
    print("+Inf in X_train:", np.isposinf(X_train).sum())
    print("-Inf in X_train:", np.isneginf(X_train).sum())

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
    
    size_layer_1 = kwargs.get('size_layer_1', 128)
    size_layer_2 = kwargs.get('size_layer_2', 64)
    dropout_rate = kwargs.get('dropout_rate', 0.05)
    batch_size = kwargs.get('batch_size', 32)
    use_focal_loss = kwargs.get('use_focal_loss', False)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(size_layer_1), BatchNormalization(), Activation('relu'), Dropout(dropout_rate),
        Dense(size_layer_2), BatchNormalization(), Activation('relu'), Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    loss = focal_loss(gamma=2., alpha=0.25) if use_focal_loss else 'binary_crossentropy'
    model.compile(optimizer=AdamW(learning_rate=cfg.learning_rate), loss=loss, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.early_stopping_patience, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=cfg.lr_reduction_factor, patience=cfg.lr_reduction_patience, min_lr=cfg.min_learning_rate)
    callbacks = [early_stopping, lr_scheduler]
    
    history = model.fit(X_train, y_train, epochs=cfg.max_epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks)
    
    logging.info(f"Number of epochs used: {len(history.epoch)}")
    model.save(cfg.model_path)

def load_tf_model(df_data, hyperparams):
    logging.info("Loading TensorFlow model...")
    logging.info(df_data.tail())

    df_input, df_output = get_dfs_input_output(df_data, cfg.output_binary_name)
    # importance_df = get_feature_importance(df_input, df_output) # Calculate SHAP importances
    # df_input = remove_highly_correlated_features(df_input, importance_df) # Remove correlated features
    test_train_data = get_test_train_data(df_input, df_output, cfg.test_size)
    
    if not os.path.exists(cfg.model_path) or not cfg.use_saved_model:
        tf.keras.backend.clear_session()
        create_tf_model(**{**test_train_data, **hyperparams})
    
    return test_train_data, tf.keras.models.load_model(cfg.model_path)
