import yfinance as yf
import pandas as pd

def compute_correlation(tickers: list):

    price_dict = {}

    for t in tickers:
        t = t.upper()

        df = yf.download(t, period="1y", progress=False)

        if df is None or len(df) == 0:
            print(f"[WARNING] Skipping {t}: No data returned.")
            continue

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if "Close" not in df:
            print(f"[WARNING] Skipping {t}: 'Close' column missing.")
            continue

        price_dict[t] = df["Close"]

    if len(price_dict) < 2:
        raise ValueError("Need at least 2 valid tickers to compute correlation.")

    df_all = pd.DataFrame(price_dict).dropna(how="all")

    if df_all.empty:
        raise ValueError("No valid overlapping data between tickers.")

    return df_all.pct_change().corr()
