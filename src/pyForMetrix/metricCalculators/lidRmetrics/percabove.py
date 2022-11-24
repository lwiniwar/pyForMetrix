import numpy as np

from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_n, basic_zmean

def percabove_pzabovemean(points):
    n_points = basic_n(points)
    mean = basic_zmean(points)
    return np.count_nonzero(points['points'][:, 2] > mean) / n_points

def percabove_pzaboveX(points, X):
    n_points = basic_n(points)
    return np.count_nonzero(points['points'][:, 2] > X) / n_points


