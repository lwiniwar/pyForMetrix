import numpy as np
import scipy.stats

from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_n, basic_zmax, basic_zmin
def lad_lad(points, dz):
    z = points['points'][:, 2]
    n = basic_n(points)
    zmax = basic_zmax(points)
    p_i = np.array([np.count_nonzero(z <= lower_bound) / (n - np.count_nonzero(z <= (lower_bound + dz)))
           for lower_bound in np.arange(0, zmax-dz, dz)]) # omit last layer, as it will result in #DIV0
    p_i = p_i[p_i > 0]
    if len(p_i) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    LAI_i = -1 * np.log(p_i) / 0.5
    LAD_i = LAI_i / dz
    return np.array([np.max(LAD_i), np.mean(LAD_i), scipy.stats.variation(LAD_i), np.min(LAD_i)])