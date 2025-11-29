import pandas as pd
import yfinance as yf

def load_api_data(ticker: str, period="5y"):
    """
    Fetch OHLCV from Yahoo Finance.
    Returns clean historical price dataframe.
    """
    df = yf.download(ticker, period=period, progress=False)
    df = df.dropna()
    df = df.astype(float, errors="ignore")
    return df


def merge_csv_if_exists(df: pd.DataFrame, csv_path: str):
    """
    Optional function: merge uploaded CSV for extended history.
    """
    try:
        csv_df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        combined = pd.concat([csv_df, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()
    except:
        return df   # fallback safely
