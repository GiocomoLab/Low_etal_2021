import numpy as np

def moving_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def nan_interp(y):
    def find(x):
        return x.nonzero()[0]
    nans = np.isnan(y)
    y[nans] = np.interp(find(nans),find(~nans),y[~nans])
    return y

def zscore(a):
    z = a - np.mean(a)
    z /= z.std()
    idx = np.abs(z) > 2

    new_a = a.copy()
    new_a[idx] = np.nan
    new_a = nan_interp(new_a)

    return new_a