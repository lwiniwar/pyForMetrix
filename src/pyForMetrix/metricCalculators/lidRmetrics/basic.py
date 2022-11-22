import numpy as np
import scipy.stats

def basic_n(points):
    return points['points'].shape[0]

def basic_zmax(points):
    return np.max(points['points'][:, 2])

def basic_zmin(points):
    return np.min(points['points'][:, 2])

def basic_zmean(points):
    return np.mean(points['points'][:, 2])

def basic_zsd(points):
    return np.std(points['points'][:, 2])

def basic_zcv(points):
    return scipy.stats.cv(points['points'][:, 2])

def basic_zskew(points):
    return scipy.stats.skew(points['points'][:, 2])

def basic_zkurt(points):
    return scipy.stats.kurtosis(points['points'][:, 2])
