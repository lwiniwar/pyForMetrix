import numpy as np

from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_n, basic_zmax
def canopydensity_zpcum(points, num_groups):
    n_points = basic_n(points)
    max_height = basic_zmax(points)
    return np.array([np.count_nonzero(points['points'][:, 2] <= (group * max_height/num_groups)) / n_points
                                            for group in range(1, num_groups)])