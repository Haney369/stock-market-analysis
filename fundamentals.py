import numpy as np
import pandas as pd
import yfinance as yf

def add_fundamentals(df: pd.DataFrame, ticker: str):
    df = df.copy()

    # Flatten ANY multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    info = yf.Ticker(ticker).info

    PE = info.get("trailingPE", np.nan)
    EV = info.get("enterpriseValue", np.nan)
    EBITDA = info.get("ebitda", np.nan)

    df["PE"] = PE
    df["EV"] = EV
    df["EBITDA"] = EBITDA

    # FairValue = price normalized relative to sector PE=25
    if PE and not np.isnan(PE):
        df["FairValue"] = df["Close"] * (25 / PE)
    else:
        df["FairValue"] = np.nan

    # Ensure FairValue is a Series, not a DataFrame
    if isinstance(df["FairValue"], pd.DataFrame):
        df["FairValue"] = df["FairValue"].iloc[:, 0]

    # Compute PriceGap safely
    df["PriceGap"] = (df["Close"] - df["FairValue"]) / df["FairValue"]

    return df
