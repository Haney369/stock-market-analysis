from fastapi import FastAPI
from pydantic import BaseModel

from data_loader import load_api_data
from technicals import compute_technicals
from fundamentals import add_fundamentals
from labels import generate_labels
from hype_momentum import compute_hype
from ensemble import train_ensemble, ensemble_predict
from correlations import compute_correlation
from visuals import (
    make_candlestick,
    make_fairvalue_plot,
    make_corr_heatmap,
    make_momentum_burst_chart
)

app = FastAPI(title="Market ML API", version="1.1")

class PredictRequest(BaseModel):
    ticker: str


@app.post("/predict")
def predict(req: PredictRequest):
    ticker = req.ticker.upper()

    df = load_api_data(ticker)
    df = compute_technicals(df)
    df = add_fundamentals(df, ticker)
    df = compute_hype(df)
    df = generate_labels(df)

    features = [
        "RSI", "MA10", "MA20", "MA50",
        "MACD", "MACD_SIGNAL", "VolSpike",
        "Volatility", "PE", "EV", "EBITDA", "PriceGap"
    ]

    models = train_ensemble(df, features)
    latest = df.iloc[-1:][features]

    growth, hype_flag = ensemble_predict(models, latest)

    return {
        "ticker": ticker,
        "predicted_growth_5d": growth,
        "hype_flag": hype_flag
    }


@app.post("/candlestick")
def candlestick(req: PredictRequest):
    df = load_api_data(req.ticker)
    df = compute_technicals(df)
    img = make_candlestick(df, req.ticker.upper())
    return {"chart": img}


@app.post("/fairvalue-plot")
def fairvalue_plot(req: PredictRequest):
    df = load_api_data(req.ticker)
    df = compute_technicals(df)
    df = add_fundamentals(df, req.ticker.upper())
    img = make_fairvalue_plot(df, req.ticker.upper())
    return {"chart": img}


@app.post("/momentum-burst")
def momentum_burst(req: PredictRequest):
    df = load_api_data(req.ticker)
    img = make_momentum_burst_chart(df, req.ticker.upper())
    return {"chart": img}


@app.post("/correlation-plot")
def correlation_plot(tickers: list[str]):
    corr_df = compute_correlation(tickers)
    img = make_corr_heatmap(corr_df)
    return {"chart": img}
