from functools import lru_cache

import numpy as np
import scipy.stats

#@lru_cache(100)
def basic_n(points):
    return points['points'].shape[0]
#@lru_cache(100)
def basic_zmax(points):
    return np.max(points['points'][:, 2])
#@lru_cache(100)
def basic_zmin(points):
    return np.min(points['points'][:, 2])
#@lru_cache(100)
def basic_zmean(points):
    if points['points'].shape[0] == 0:
        return np.nan
    return np.mean(points['points'][:, 2])

def basic_zsd(points):
    return np.std(points['points'][:, 2])

def basic_zcv(points):
    return scipy.stats.variation(points['points'][:, 2])

def basic_zskew(points):
    return scipy.stats.skew(points['points'][:, 2])

def basic_zkurt(points):
    return scipy.stats.kurtosis(points['points'][:, 2])

