import pandas as pd
import os
import config

from utils.yahoo_downloader import YahooDownloader
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
    df[config.stats]
    df['change_-10~0_min'] = df['close'] / df['close_-10~0_min']
    df['change_-10~0_max'] = df['close'] / df['close_-10~0_max']
    df[config.target_feature] = df['high_100_d'] > 5

    pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.shape)

    return df.dropna()

def plot_correlations(df: pd.DataFrame, folder_path: str, file_name: str) -> None:
    """ Save a pdf plot of correlations between features """
    df = df[::-1]  # reverse temporality
    corr_df = df.corr()
    plt.figure(figsize=(150,100))
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
    feature_importances.plot(figsize=(25,15), kind='barh', title='Feature importances')
    plt.subplots_adjust(left=0.3)
    plt.savefig(folder_path + file_name)
    plt.close()
    
def main():
    df = get_yahoo_data(
        folder_path='./db/', file_name='ohlcv.pkl',
        start_date=config.start_date, end_date=config.end_date,
        tickers=config.tickers)
    df = df.loc[df['tic'] == 'FR.PA']

    df = add_stats(df)

    # plot_correlations(df=df, folder_path='./plots/', file_name='correlations.pdf')
    plot_feature_importances(
        df=df,
        X_features=config.X_features, y_feature=config.target_feature,
        folder_path='./plots/', file_name='feature_importances.pdf')

if __name__ == "__main__":
    main()
