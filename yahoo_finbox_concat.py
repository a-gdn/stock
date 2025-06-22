import pandas as pd

# Load the two DataFrames
finbox_df = pd.read_pickle('./db/fundamentals/finbox_fundamentals.pkl')
ohlcv_df = pd.read_pickle('./db/ohlcv/ohlcv_ntickers_593_2000-08-01_to_2024-06-15.pkl')

# Keep only common tickers (based on level=1 of MultiIndex columns)
ohlcv_tickers = set(ohlcv_df.columns.get_level_values(1))
finbox_tickers = set(finbox_df.columns.get_level_values(1))
common_tickers = ohlcv_tickers & finbox_tickers

print(f"Number of Yahoo (ohlcv) tickers: {len(ohlcv_tickers)}")
print(f"Number of Finbox tickers: {len(finbox_tickers)}")
print(f"Number of common tickers: {len(common_tickers)}")

ohlcv_df = ohlcv_df.loc[:, ohlcv_df.columns.get_level_values(1).isin(common_tickers)]
finbox_df = finbox_df.loc[:, finbox_df.columns.get_level_values(1).isin(common_tickers)]

# Reindex finbox_df to daily index of ohlcv_df and forward fill
finbox_aligned = finbox_df.reindex(ohlcv_df.index).ffill()

# Concatenate along the columns
merged_df = pd.concat([ohlcv_df, finbox_aligned], axis=1)

print(merged_df.info())
print(merged_df.tail(10))

merged_df.to_pickle("./db/merged_ohlcv_fundamentals.pkl")