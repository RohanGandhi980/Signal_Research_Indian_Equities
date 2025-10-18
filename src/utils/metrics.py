import pandas as pd
import numpy as np
from scipy.stats import norm

def rms(y, ywhat):
    return float(np.sqrt(np.mean((y-ywhat)**2)))

def mae(y,ywhat):
    return float(np.mean(np.abs(y-ywhat)))

def smape(y,ywhat):
    denom = (np.abs(y)+np.abs(ywhat)) /2.0+1e-12
    return float(np.mean(np.abs(y-ywhat)/denom))

def dm_test(e1,e2,h=1):
    #diebold mariano for equal predictive accuracy
    d=(e1**2 - e2**2)
    dbar = np.mean(d)

    #hac variance approx for horizon h
    gamma = [np.cov(d[:-k],d[k:])[0,1] for k in range(1,h)]
    var = (np.var(d, ddof=1)+ 2*sum(gamma)) /len(d)
    stat = dbar/(np.sqrt(var)+1e-12)
    pval = 2*(1-norm.cdf(abs(stat)))
    return float(stat), float(pval)
