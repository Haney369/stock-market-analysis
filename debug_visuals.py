from data_loader import load_api_data
from visuals import (
    make_candlestick,
    make_fairvalue_plot,
    make_corr_heatmap,
    make_momentum_burst_chart,
    SHOW_BACKEND_PLOTS
)

# Enable backend visuals
SHOW_BACKEND_PLOTS = True

ticker = "AMD"   # change as needed

df = load_api_data(ticker)

# --- Candlestick Chart ---
print("Generating Candlestick Chart...")
make_candlestick(df, ticker)

# --- Fair Value Plot ---
from fundamentals import add_fundamentals
df_fund = add_fundamentals(df.copy(), ticker)
print("Generating Fair Value Chart...")
make_fairvalue_plot(df_fund, ticker)

# --- Momentum Burst Chart ---
print("Generating Momentum Burst Chart...")
make_momentum_burst_chart(df, ticker)

# --- Correlation Heatmap Example ---
from correlations import compute_correlation
corr = compute_correlation(["AMD", "NVDA", "MSFT", "AAPL"])
print("Generating Correlation Heatmap...")
make_corr_heatmap(corr)
