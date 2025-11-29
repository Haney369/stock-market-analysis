import pandas as pd

def generate_labels(df: pd.DataFrame, fwd_days=5):
    df = df.copy()
    df["FuturePrice"] = df["Close"].shift(-fwd_days)
    df["FwdReturn"] = (df["FuturePrice"] - df["Close"]) / df["Close"]
    return df.dropna()
