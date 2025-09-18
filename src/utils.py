
import pandas as pd

def load_series(csv_path, date_col, target_col):
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df[[date_col, target_col]].dropna().sort_values(date_col).reset_index(drop=True)
    return df

def train_test_split_ts(df, test_size):
    return df.iloc[:-test_size, :].copy(), df.iloc[-test_size:, :].copy()
