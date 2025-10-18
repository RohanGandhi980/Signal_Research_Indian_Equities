import pandas as pd
from src.utils.metrics import rmse, mape, directional_accuracy
from src.backtest.signals import positions_from_forecast, pnl
from src.backtest.metrics import sharpe, equity_curve, cagr

def evaluate(close:pd.Series, forecast:pd.Series, hold:int=1, rf:float=0.0):
    # Forecast metrics
    f_rmse = rmse(close, forecast)
    f_mape = mape(close, forecast)
    f_dir  = directional_accuracy(close, forecast)

    # Strategy metrics
    positions = positions_from_forecast(close, forecast, hold=hold)
    daily = pnl(close, positions)
    curve = equity_curve(daily, init=1.0)
    m_sharpe = sharpe(daily, rf=rf)
    m_cagr   = cagr(curve)

    return {
        "forecast": {"RMSE": f_rmse, "MAPE": f_mape, "DirAcc": f_dir},
        "strategy": {"Sharpe": m_sharpe, "CAGR": m_cagr},
        "equity_curve": curve,
        "daily_returns": daily,
        "positions": positions
    }
