import lmoments3
import numpy as np


def Lmoments_moments(points):
    try:
        res = lmoments3.lmom_ratios(points['points'][:, 2], 4)
    except:
        res = np.array([np.nan, np.nan, np.nan, np.nan])
    return res

def Lmoments_coefficients(points):
    mom = Lmoments_moments(points)
    L_CV = mom[1]/mom[0]
    L_skew = mom[2]/mom[1]
    L_kurt = mom[3]/mom[2]
    return np.array([L_skew, L_kurt, L_CV])