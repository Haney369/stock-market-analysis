import ta
import pandas as pd

def compute_technicals(df: pd.DataFrame):
    df = df.copy()

    # RSI, MAs, MACD, Volatility, Volume Spike
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14, fillna=True)
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df["Volatility"] = df["Close"].pct_change().rolling(10).std()
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df.fillna(method="bfill").fillna(method="ffill")
