import pandas as pd
import os
import config
from utils.yahoodownloader import YahooDownloader
from stockstats import wrap
# from pyspark.sql import SparkSession

def get_yahoo_data(folder_path, file_name, start_date, end_date, tickers):
    file_path = folder_path + '/' + file_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.isfile(file_path):
        df = pd.read_pickle(file_path)
    else:
        df = YahooDownloader(start_date, end_date, tickers).fetch_data()        
        df.to_pickle(file_path)

    return df

def main():
    df = get_yahoo_data('./db', 'ohlcv.pkl', config.start_date, config.end_date, config.tickers)
    print(df.head())
    print(df.shape)

    df = df.loc[df['tic'] == 'FR.PA']
    print(df.head())
    print(df.shape)

    df = wrap(df)
    print(df.head())
    print(df.shape)

    df['ups'], df['downs'] = df['change'] > 0, df['change'] < 0 

    df = df[[
        'change',
        'rsi', 'rsi_6',
        'log-ret',
        'ups_10_c', 'downs_10_c',
        'close_-2~0_min', 'close_-2~0_max',
        'stochrsi', 'stochrsi_6',
        'wt1', 'wt2',
        'trix', 'middle_10_trix',
        'tema', 'middle_10_tema',
        'vr', 'vr_6',
        'wr', 'wr_6',
        'cci', 'cci_6',
        'atr', 'atr_5',
        'supertrend', 'supertrend_ub', 'supertrend_lb',
        'dma',
        'pdi', 'mdi', 'dx', 'adx', 'adxr',
        'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3',
        'middle',
        'boll', 'boll_ub', 'boll_lb',
        'macd', 'macds', 'macdh',
        'ppo', 'ppos', 'ppoh',
        'vwma', 'vwma_6',
        'chop', 'chop_6',
        'mfi', 'mfi_6',
    ]]
    print(df.head())
    print(df.shape)

    df = df.dropna()
    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    main()
