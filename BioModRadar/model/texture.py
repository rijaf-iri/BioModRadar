import numpy as np

def compute_texture_jatau(x, min_gates):
    c0 = x.shape[0] // 2
    c1 = x.shape[1] // 2
    c = x[c0, c1]
    if np.isnan(c): return np.nan
    y = x.reshape(-1)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < min_gates: return np.nan
    return np.sum(np.abs(c - y))/(n - 1)

def compute_texture_ndimage_jatau(x, min_gates):
    c = x[x.shape[0] // 2]
    if np.isnan(c): return np.nan
    y = x[~np.isnan(x)]
    n = len(y)
    if n < min_gates: return np.nan
    return np.sum(np.abs(c - y))/(n - 1)

def compute_texture_ndimage_stdev(x, min_gates):
    c = x[x.shape[0] // 2]
    if np.isnan(c): return np.nan
    y = x[~np.isnan(x)]
    if len(y) < min_gates: return np.nan
    return np.std(y)

def compute_ndimage_mean(x, min_gates):
    c = x[x.shape[0] // 2]
    if np.isnan(c): return np.nan
    y = x[~np.isnan(x)]
    if len(y) < min_gates: return np.nan
    return np.mean(y)

def compute_ndimage_median(x, min_gates):
    c = x[x.shape[0] // 2]
    if np.isnan(c): return np.nan
    y = x[~np.isnan(x)]
    if len(y) < min_gates: return np.nan
    return np.median(y)
