from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
from src.features.engineering import build_features
from src.models.train import train_sarima, train_lstm
from src.models.forecast import predict_next_sarima, predict_next_lstm
from src.utils.logger import get_logger

app = FastAPI(title="Signal Research API")
log = get_logger(__name__)

class TickerReq(BaseModel):
    ticker: str
    start_date: Optional[str] = "2020-01-01"
    end_date: Optional[str] = None

class ForecastReq(BaseModel):
    ticker: str
    model: str = "sarima"

@app.post("/ingest")
def ingest(req: TickerReq):
    end_date = req.end_date if req.end_date not in [None, "string"] else datetime.today().strftime("%Y-%m-%d")
    df = build_features(req.ticker, req.start_date, end_date, target_col="close")
    return {"status": "ok", "ticker": req.ticker, "rows": len(df)}

@app.post("/train")
def train(req: TickerReq):
    ticker = req.ticker.upper()
    start_date = req.start_date or "2020-01-01"
    end_date = req.end_date if req.end_date not in [None, "string"] else datetime.today().strftime("%Y-%m-%d")

    log.info(f"[API] Training models for {ticker} from {start_date} to {end_date}")
    build_features(ticker, start_date=start_date, end_date=end_date)
    s = train_sarima(ticker)
    l = train_lstm(ticker)

    return {"ticker": ticker, "sarima": "trained", "lstm": "trained", "range": f"{start_date} → {end_date}"}

@app.post("/forecast")
def forecast(req: ForecastReq):
    ticker = req.ticker.upper()
    next_day = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"[API] Forecasting {ticker} for next day ({next_day}) using {req.model.upper()}")

    if req.model.lower() == "sarima":
        pred = predict_next_sarima(ticker)
    else:
        pred = predict_next_lstm(ticker)

    return {"ticker": ticker, "model": req.model, "forecast_date": next_day, "predicted_close": pred}
