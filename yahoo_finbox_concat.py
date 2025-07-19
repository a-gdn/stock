import pandas as pd
import numpy as np

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

# Align dates
finbox_aligned = finbox_df.reindex(ohlcv_df.index)

# Concatenate along the columns
merged_df = pd.concat([ohlcv_df, finbox_aligned], axis="columns")

def fill_market_cap(uncomplete_market_cap, price):
    # Estimate shares outstanding
    shares = uncomplete_market_cap / price
    shares.ffill(inplace=True)

    return price * shares

def fill_pe_ltm(uncomplete_pe_ltm, price):
    # Estimate earnings per share (EPS)
    eps = price / uncomplete_pe_ltm
    eps = eps.replace([np.inf, -np.inf], np.nan)
    eps.ffill(inplace=True)

    return price / eps

def fill_price_to_book(uncomplete_price_to_book, price):
    # Estimate book value per share (bvps)
    bvps = price / uncomplete_price_to_book
    bvps = bvps.replace([np.inf, -np.inf], np.nan)
    bvps.ffill(inplace=True)

    return price / bvps

# For some fundamentals, estimate missing data based on price
for ticker in common_tickers:
    price = merged_df[('Close', ticker)].replace(0, np.nan)

    merged_df[('marketcap', ticker)] = fill_market_cap(merged_df[('marketcap', ticker)], price)
    merged_df[('pe_ltm', ticker)] = fill_pe_ltm(merged_df[('pe_ltm', ticker)], price)
    merged_df[('price_to_book', ticker)] = fill_price_to_book(merged_df[('price_to_book', ticker)], price)

# For the rest of the fundamentals, forward fill missing values
merged_df.ffill(inplace=True)

merged_df.info()

merged_df.to_pickle("./db/merged_ohlcv_fundamentals.pkl")