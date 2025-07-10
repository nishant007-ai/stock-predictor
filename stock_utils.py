# stock_utils.py
import pandas as pd

def clean_columns(df):
    """Flatten columns in case of MultiIndex from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_popular_tickers():
    return ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'RELIANCE.NS', 'TATASTEEL.NS']
