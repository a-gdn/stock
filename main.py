import pandas as pd
import os

import config as cfg
from utils.yahoo_downloader import YahooDownloader
import utils.helper_functions as help_fn

from stockstats import wrap

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif


def get_yahoo_data(folder_path: str, file_name: str, start_date: str, end_date: str, tickers: list[str]) -> pd.DataFrame:
    """ Save a db with ohlcv + day + tickers data and return df """
    file_path = folder_path + file_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.isfile(file_path):
        df = pd.read_pickle(file_path)
    else:
        df = YahooDownloader(start_date, end_date, tickers).fetch_data()        
        df.to_pickle(file_path)

    return df

def add_stats(df: pd.DataFrame) -> pd.DataFrame:
    """" Add stockstats data to df """
    df = wrap(df) # Convert to stockstats legible format

    df['ups'] = df['change'] > 0
    df['downs'] = df['change'] < 0
    df[cfg.stats]
    df['change_-10~0_min'] = df['close'] / df['close_-10~0_min']
    df['change_-10~0_max'] = df['close'] / df['close_-10~0_max']
    df[cfg.target_feature] = df[::-1]['close'].rolling(cfg.target_days).max() / df['close']
    df[cfg.target_feature_class] = df[cfg.target_feature] > 1 + cfg.target_percent_increase

    print(df.head())
    print(df.shape)

    return df.dropna()

def plot_correlations(df: pd.DataFrame, folder_path: str, file_name: str) -> None:
    """ Save a pdf plot of correlations between features """
    df = df[::-1]  # reverse temporality
    corr_df = df.corr().round(2)
    plt.figure(figsize=(70,40))
    sns.heatmap(corr_df, annot=True)
    plt.subplots_adjust(left=0.2, bottom=0.35)
    plt.savefig(folder_path + file_name)
    plt.close()

def plot_feature_importances(df: pd.DataFrame, X_features: list, y_feature: str, folder_path: str, file_name: str) -> None:
    """ Save a pdf plot of feature importances on injuries """
    X_df = df[X_features]
    y_df = df[y_feature]

    feature_importances = mutual_info_classif(X_df, y_df, random_state=0, n_neighbors=3, discrete_features='auto')
    feature_importances = pd.Series(feature_importances, X_df.columns)
    feature_importances.plot(figsize=(20,15), kind='barh', title='Feature importances')
    plt.subplots_adjust(left=0.3)
    plt.savefig(folder_path + file_name)
    plt.close()

def add_supports_resistances(df: pd.DataFrame) -> pd.DataFrame:
    df['rolling_min'] = help_fn.get_rolling_min(df, 'close', cfg.target_days)
    df['rolling_max'] = help_fn.get_rolling_max(df, 'close', cfg.target_days)

    df['pivot'] = df.apply(lambda row: help_fn.get_pivot(row['rolling_max'], row['rolling_min'], row['close']), axis = 1)
    
    df['support1'] = df.apply(lambda row: help_fn.get_support1(row['pivot'], row['rolling_max']), axis = 1)
    df['support2'] = df.apply(lambda row: help_fn.get_support2(row['pivot'], row['rolling_max'], row['rolling_min']), axis = 1)
    df['resistance1'] = df.apply(lambda row: help_fn.get_resistance1(row['pivot'], row['rolling_min']), axis = 1)
    df['resistance2'] = df.apply(lambda row: help_fn.get_resistance2(row['pivot'], row['rolling_max'], row['rolling_min']), axis = 1)
    
    return df

def main():
    df = get_yahoo_data(
        folder_path='./db/', file_name='ohlcv.pkl',
        start_date=cfg.start_date, end_date=cfg.end_date,
        tickers=cfg.tickers)
    df = df.loc[df['tic'] == 'FR.PA']

    df = add_supports_resistances(df)

    print(df.head())

    # df = add_stats(df)
    # plot_correlations(df=df, folder_path='./plots/', file_name='correlations.pdf')
    # plot_feature_importances(
    #     df=df,
    #     X_features=cfg.X_features, y_feature=cfg.target_feature_class,
    #     folder_path='./plots/', file_name='feature_importances.pdf')

if __name__ == "__main__":
    main()
