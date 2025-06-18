import pandas as pd

# Load the two DataFrames
finbox_df = pd.read_pickle('./db/fundamentals/finbox_fundamentals.pkl')
ohlcv_df = pd.read_pickle('./db/ohlcv/ohlcv_ntickers_593_2000-08-01_to_2024-06-15.pkl')

# Reindex finbox_df to daily index of ohlcv_df and forward fill
finbox_aligned = finbox_df.reindex(ohlcv_df.index).ffill()

# Concatenate along the columns
merged_df = pd.concat([ohlcv_df, finbox_aligned], axis=1)
merged_df_tail = merged_df.tail(10)
print(merged_df_tail)

merged_df.to_pickle("./db/merged_ohlcv_fundamentals.pkl")