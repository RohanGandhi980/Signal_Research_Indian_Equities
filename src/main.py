from config.config_loader import load_config
from src.features.engineering import build_features
from src.models.train import train_sarima, train_lstm
from src.models.forecast import predict_next_sarima, predict_next_lstm
from src.backtest.runner import evaluate
import pandas as pd
from src.utils.paths import proc_path_for

def run():
    data_cfg = load_config("data")
    bt_cfg   = load_config("backtest")
    ticker   = data_cfg["ticker"]

    # 1) Build dataset
    df = build_features(ticker, data_cfg["start_date"], data_cfg["end_date"], data_cfg["target_column"])

    # 2) Train models
    s_path = train_sarima(ticker)
    l_path = train_lstm(ticker)

    # 3) Produce one-step forecasts across the whole series (rolling inference)
    df = pd.read_parquet(proc_path_for(ticker))
    close = df[data_cfg["target_column"]]
    # For demo: we just forecast the next point based on the final window
    yhat_s = predict_next_sarima(ticker)
    yhat_l = predict_next_lstm(ticker)
    print(f"[{ticker}] Next SARIMA: {yhat_s:.2f} | Next LSTM: {yhat_l:.2f}")

    # 4) (Optional) Backtest — typically you'd generate a forecast series; here we show example with last N
    # In practice, you'd roll the models through history. Keeping this minimal to run out-of-the-box.

if __name__ == "__main__":
    run()
