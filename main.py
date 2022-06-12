import pandas as pd
import os
import config
from utils.yahoodownloader import YahooDownloader
from pyspark.sql import SparkSession

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
    pandas_df = get_yahoo_data('./db', 'ohlcv.pkl', config.start_date, config.end_date, config.tickers)

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(pandas_df)

    print(df.show(10))

if __name__ == "__main__":
    main()
