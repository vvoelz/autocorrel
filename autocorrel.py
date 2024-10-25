# Tools and helper function for autocorrelation functions
# NOTE: requires python3

import numpy as np

def autocorr(x):
    tau = x.size
    mu = x.mean()
    g = np.correlate(x, x, mode='full')[tau-1:]
    n = np.arange(tau,0,-1)
    return g/n
