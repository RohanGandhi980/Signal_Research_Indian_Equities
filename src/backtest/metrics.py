import numpy as np, pandas as pd

def sharpe(daily, rf=0.0):
    r = np.array(daily) - rf/252
    return float(np.mean(r)/(np.std(r)+1e-9)*np.sqrt(252))

def equity_curve(daily, init=1.0):
    return pd.Series(daily).fillna(0).add(1).cumprod()*init

def cagr(curve: pd.Series):
    if len(curve)==0: return 0.0
    years = len(curve)/252
    return float((curve.iloc[-1] / curve.iloc[0])**(1/years) - 1)
