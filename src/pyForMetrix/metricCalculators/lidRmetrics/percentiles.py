import numpy as np
def percentiles_z(points, percentiles):
    return np.percentile(points['points'][:, 2], percentiles)