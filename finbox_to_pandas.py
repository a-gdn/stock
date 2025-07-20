import pandas as pd
import os
import json
from pathlib import Path
import utils.helper_functions as hf

def add_pct_change_columns(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    all_pct_change_dfs = []

    for col in df.columns:
        fundamental, ticker = col
        series = df[col]
        known_indices = series[series.notna()].index

        for period in periods:
            new_col = (f"{fundamental}_var_{period}", ticker)
            pct_changes = pd.Series(index=series.index, dtype="float64")

            for i in range(period, len(known_indices)):
                current_idx = known_indices[i]
                prev_idx = known_indices[i - period]
                current_val = series.loc[current_idx]
                prev_val = series.loc[prev_idx]

                # Avoid division by zero
                if prev_val == 0:
                    pct_changes.loc[current_idx] = pd.NA
                else:
                    pct_changes.loc[current_idx] = (current_val / prev_val) - 1

            pct_change_df = pct_changes.to_frame(name=new_col)
            all_pct_change_dfs.append(pct_change_df)

    pct_change_df_combined = pd.concat(all_pct_change_dfs, axis=1)
    combined_df = pd.concat([df, pct_change_df_combined], axis=1)
    combined_df = combined_df.sort_index(axis=1, level=[0, 1])

    return combined_df

def extract_finbox_ticker_from_path(file_path: Path, base_directory: Path) -> str:
    relative_path = file_path.relative_to(base_directory).with_suffix('')
    ticker_str = str(relative_path).replace(os.sep, ':')
    ticker_parts = ticker_str.split('_')
    finbox_ticker = ticker_parts[-1]
    yahoo_ticker = hf.finbox_to_yahoo(finbox_ticker)
    return yahoo_ticker

def remove_tickers_with_all_nan(df: pd.DataFrame) -> pd.DataFrame:
    tickers = df.columns.get_level_values('Ticker').unique()
    tickers_to_drop = set()

    for ticker in tickers:
        ticker_df = df.xs(ticker, axis=1, level='Ticker', drop_level=False)
        
        # If any column for this ticker is fully NaN → mark for drop
        if ticker_df.isna().all().any():
            tickers_to_drop.add(ticker)

    print(f"\nTickers with at least one column fully NaN (to be removed):")
    for ticker in sorted(tickers_to_drop):
        print(ticker)

    print(f"Number of tickers removed due to a nan column: {len(tickers_to_drop)}")

    df = df.drop(columns=tickers_to_drop, level='Ticker')
    return df

def create_fundamentals_dataframe(directory_path: str) -> pd.DataFrame:
    data_path = Path(directory_path)
    if not data_path.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
        return pd.DataFrame()

    all_series = []
    tickers_to_skip = set()

    print(f"Scanning files in '{data_path.resolve()}'...")

    for file_path in data_path.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            chart_data = content.get('data', {}).get('company', {}).get('glossary', {}).get('chart', {})
            if not chart_data:
                print(f"Warning: Skipping '{file_path.name}'. No 'chart' data found.")
                continue

            ticker = extract_finbox_ticker_from_path(file_path, data_path)
            metric = chart_data.get('metrics', [None])[0]

            series_info = chart_data.get('data', [{}])[0]
            dates = series_info.get('period_dates')
            values = series_info.get('values')

            if not all([ticker, metric, dates, values]):
                print(f"Warning: Skipping '{file_path.name}' due to incomplete data.")
                continue

            processed_values = pd.to_numeric(values, errors='coerce')
            series = pd.Series(
                data=processed_values,
                index=pd.to_datetime(dates),
                name=(metric, ticker)
            )

            if series.index.duplicated().any():
                print(f"Warning: {series.name} has duplicated dates. Will exclude ticker '{ticker}' entirely.")
                tickers_to_skip.add(ticker)

            all_series.append(series)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error processing '{file_path.name}': {e}")
        except Exception as e:
            print(f"Unexpected error with file '{file_path.name}': {e}")

    if not all_series:
        print("No valid data found to create a DataFrame.")
        return pd.DataFrame()

    clean_series = [s for s in all_series if s.name[1] not in tickers_to_skip]

    print("\nCombining all data into a single DataFrame...")
    final_df = pd.concat(clean_series, axis=1)

    # Sort the DataFrame
    final_df.sort_index(axis='index', inplace=True) # Sort rows by date
    final_df.sort_index(axis='columns', level=[0, 1], inplace=True) # Sort columns by fundamental, then by ticker
    
    # Set names for the DataFrame
    final_df.columns.names = ['Fundamental', 'Ticker']
    final_df.index.name = 'Date'
    
    # final_df.ffill(inplace=True)

    final_df = remove_tickers_with_all_nan(final_df)
    final_df = add_pct_change_columns(final_df, periods=[1, 2, 4])
    final_df.replace([float('inf'), -float('inf')], [1e6, -1e6], inplace=True)

    print("Processing complete.")
    return final_df

if __name__ == "__main__":
    fundamentals_directory = './db/fundamentals'
    master_df = create_fundamentals_dataframe(fundamentals_directory)

    if not master_df.empty:
        print("\n--- DataFrame Info ---")
        master_df.info()

        print("\n--- DataFrame Tail ---")
        print(master_df.tail())

        print("\n--- Example: Accessing A2A's Current Ratio ---")
        print(master_df['current_ratio']['A2A.MI'].tail())

        pickle_file = os.path.join(fundamentals_directory, 'finbox_fundamentals.pkl')
        master_df.to_pickle(pickle_file)

        print(f"master_df successfully saved to {pickle_file}")