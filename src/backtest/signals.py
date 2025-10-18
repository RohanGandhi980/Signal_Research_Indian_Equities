import pandas as pd

def positions_from_forecast(close:pd.Series, forecast:pd.Series, hold:int=1):
    pos = (forecast > close).astype(int)  # long if forecast up; else flat
    if hold>1: pos = pos.rolling(hold).max()
    return pos.reindex(close.index).fillna(0)

def pnl(close:pd.Series, positions:pd.Series):
    ret = close.pct_change().fillna(0.0)
    return positions.shift(1).fillna(0.0) * ret
