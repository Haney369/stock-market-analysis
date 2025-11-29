import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import io

# Toggle: show charts locally or not
SHOW_BACKEND_PLOTS = True


def fig_to_base64(fig):
    if SHOW_BACKEND_PLOTS:
        plt.show()   # show backend visuals during development

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# -------------------------------
# 1. Candlestick Chart
# -------------------------------
def make_candlestick(df, ticker):

    # Make sure df is not empty
    if df is None or len(df) == 0:
        raise ValueError(f"No data available for {ticker}. DataFrame is empty.")

    df = df.copy()

    # If YFinance returns MultiIndex columns (sometimes happens)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Check if any OHLC columns are missing
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    # Force-cast numeric OHLC
    for col in required_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            raise TypeError(f"Failed casting column {col} to numeric: {e}")

    # Drop rows missing key values
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if len(df) < 10:
        raise ValueError(f"Not enough valid data to plot candlestick for {ticker}.")

    # Generate plot
    fig, ax = mpf.plot(
        df.tail(120),
        type="candle",
        volume=True,
        style="yahoo",
        mav=(10, 20, 50),
        returnfig=True
    )

    return fig_to_base64(fig)

# -------------------------------
# 2. Fair Value vs Market Price
# -------------------------------
def make_fairvalue_plot(df, ticker):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Market Price", color="blue")
    plt.plot(df.index, df["FairValue"], label="Fair Value", color="orange")
    plt.grid(True, alpha=0.3)
    plt.title(f"{ticker} — Fair Value vs Market Price")
    plt.legend()
    return fig_to_base64(fig)


# -------------------------------
# 3. Correlation Heatmap
# -------------------------------
def make_corr_heatmap(corr_df):
    # Validate input
    if corr_df is None or not hasattr(corr_df, "shape"):
        raise ValueError("Correlation matrix is invalid (None or wrong type).")

    if len(corr_df.shape) != 2:
        raise ValueError(f"Correlation matrix must be 2D. Got shape={corr_df.shape}")

    if corr_df.shape[0] < 2:
        raise ValueError("Correlation matrix must be at least 2x2.")

    fig = plt.figure(figsize=(8,6))
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1,
        vmax=1
    )
    plt.title("Stock Correlation Matrix")

    return fig_to_base64(fig)

# -------------------------------
# 4. Momentum Burst Chart
# -------------------------------
def make_momentum_burst_chart(df, ticker):
    df = df.copy()

    df["Momentum"] = df["Close"].pct_change(3)
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Burst"] = (df["Momentum"] > 0.07) & (df["VolSpike"] > 1.8)

    fig = plt.figure(figsize=(12,6))

    plt.plot(df.index, df["Momentum"], label="Momentum (3-day)", color="blue", linewidth=2)
    plt.plot(df.index, df["VolSpike"], label="Volume Spike", color="orange", alpha=0.7)

    burst_points = df[df["Burst"] == True]
    plt.scatter(burst_points.index, burst_points["Momentum"],
                color="red", s=80, label="Momentum Burst 🔥")

    plt.title(f"{ticker} — Momentum Burst Chart")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return fig_to_base64(fig)
