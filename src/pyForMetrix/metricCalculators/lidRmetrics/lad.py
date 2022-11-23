import numpy as np
import scipy.stats

from basic import basic_n, basic_zmax, basic_zmin
def lad_lad(points, dz):
    xyz = points['points']
    n = basic_n(xyz)
    zmax = basic_zmax(xyz)
    p_i = [np.count_nonzero(xyz[:, 2] <= lower_bound) / (n - np.count_nonzero(xyz[:, 2] <= (lower_bound + dz)))
           for lower_bound in range(0, zmax, dz)]
    LAI_i = -1 * np.log(p_i) / 0.5
    LAD_i = LAI_i / dz
    return np.array([np.max(LAD_i), np.mean(LAD_i), scipy.stats.variation(LAD_i), np.min(LAD_i)])