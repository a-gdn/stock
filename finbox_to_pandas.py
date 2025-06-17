import pandas as pd
import os
import json
from pathlib import Path
import pickle


def create_fundamentals_dataframe(directory_path: str) -> pd.DataFrame:
    """
    Reads all JSON fundamental files from a directory, processes them,
    and returns a single multi-indexed Pandas DataFrame.

    Args:
        directory_path: The path to the directory containing the JSON files.

    Returns:
        A Pandas DataFrame where:
        - The index is the 'period_date'.
        - Columns are a MultiIndex with 'Ticker' at the top level
          and 'Fundamental' at the sub-level.
    """
    data_path = Path(directory_path)
    if not data_path.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
        return pd.DataFrame()

    all_series = []

    print(f"Scanning files in '{data_path.resolve()}'...")
    # Iterate over all json files in the specified directory
    for file_path in data_path.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # --- Safely extract data from the JSON structure ---
            # Using .get() prevents errors if keys are missing
            chart_data = content.get('data', {}).get('company', {}).get('glossary', {}).get('chart', {})
            
            if not chart_data:
                print(f"Warning: Skipping '{file_path.name}'. No 'chart' data found.")
                continue

            ticker = chart_data.get('full_tickers', [None])[0]
            metric = chart_data.get('metrics', [None])[0]
            
            # The actual time series is nested in another 'data' list
            series_info = chart_data.get('data', [{}])[0]
            dates = series_info.get('period_dates')
            values = series_info.get('values')

            # --- Validate extracted data ---
            if not all([ticker, metric, dates, values]):
                print(f"Warning: Skipping '{file_path.name}' due to incomplete data (ticker, metric, dates, or values).")
                continue

            # --- Create a Pandas Series for the current file ---
            # Coerce non-numeric values (like 'NA' or null) to NaN
            processed_values = pd.to_numeric(values, errors='coerce')

            # The series name is a tuple, which will become the multi-level column
            series = pd.Series(
                data=processed_values,
                index=pd.to_datetime(dates),
                name=(ticker, metric)
            )

            if series.index.duplicated().any():
                print(f"Warning: {series.name} has duplicated dates. Skipping.")
                continue

            all_series.append(series)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error processing '{file_path.name}': Invalid format or missing key. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with file '{file_path.name}': {e}")
    
    if not all_series:
        print("No valid data found to create a DataFrame.")
        return pd.DataFrame()

    for s in all_series:
        duplicates = s.index[s.index.duplicated()]
        if not duplicates.empty:
            print(f"{s.name} has duplicated dates:\n{duplicates}")

    # --- Combine all individual series into a single DataFrame ---
    # axis=1 aligns the series as columns based on their index (dates)
    print("\nCombining all data into a single DataFrame...")
    final_df = pd.concat(all_series, axis=1)

    # --- Polish the DataFrame for final output ---
    # Sort by date
    final_df.sort_index(inplace=True)
    # Sort columns first by Ticker (level 0), then by Fundamental (level 1)
    final_df.sort_index(axis=1, level=[0, 1], inplace=True)
    # Name the column levels for clarity
    final_df.columns.names = ['Ticker', 'Fundamental']
    
    print("Processing complete.")
    return final_df

# 2. Specify the directory and run the main function
# Replace './db/fundamentals' with the actual path to your files.
fundamentals_directory = './db/fundamentals'
master_df = create_fundamentals_dataframe(fundamentals_directory)

# 3. Display the results
if not master_df.empty:
    print("\n--- DataFrame Info ---")
    master_df.info()
    
    print("\n--- DataFrame Head ---")
    print(master_df.head())

    # Example of how to access data
    print("\n--- Example: Accessing A2A's Current Ratio ---")
    print(master_df['BIT:A2A']['current_ratio'].head())

    pickle_file = os.path.join(fundamentals_directory, 'finbox_fundamentals.pkl')
    master_df.to_pickle(pickle_file)

    print(f"master_df successfully saved to {pickle_file}")