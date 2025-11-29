import pandas as pd

def compute_hype(df: pd.DataFrame):
    """
    Computes hype flag using:
    - 3-day momentum burst
    - abnormal volume spike
    - price deviation vs fair value
    """

    df = df.copy()
    df["Momentum"] = df["Close"].pct_change(3)
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["PriceGap"] = (df["Close"] - df["FairValue"]) / df["FairValue"]

    # Hype is triggered when multiple signals align
    df["Hype"] = (
        (df["Momentum"].abs() > 0.07) &     # >=7% move in 3 days
        (df["VolSpike"] > 1.8) &           # 80% more volume
        (df["PriceGap"].abs() > 0.10)      # 10% deviation from fair
    ).astype(int)

    return df
