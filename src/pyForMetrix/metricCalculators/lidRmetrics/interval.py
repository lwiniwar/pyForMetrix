import numpy as np

from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_n

def interval_p_below(points, threshold):
    n = basic_n(points)
    xyz = points['points']
    return np.count_nonzero(xyz[:, 2] < threshold) / n