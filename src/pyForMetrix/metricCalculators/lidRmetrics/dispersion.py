import numpy as np
import scipy.stats
from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_zmean

def dispersion_ziqr(points):
    return np.diff(np.quantile(points['points'][:, 2], [0.25, 0.75]))[0]

def dispersion_zMADmean(points):
    mean = basic_zmean(points)
    if np.isnan(mean):
        return np.nan
    return np.mean(np.abs(points['points'][:, 2] - mean))

def dispersion_zMADmedian(points):
    median = np.median(points['points'][:, 2])
    return np.mean(np.abs(points['points'][:, 2] - median))

def dispersion_CRR(points):
    mean = basic_zmean(points)
    ptp = np.ptp(points['points'][:, 2])
    min = np.min(points['points'][:, 2])
    return (mean - min)/(ptp)

def dispersion_zentropy(points, binsize=1):
    hist = np.histogram(points['points'][:, 2],
                        bins=np.arange(np.min(points['points'][:, 2]), np.max(points['points'][:, 2]) + binsize, binsize),
                        density=True)[0]
    hist = hist.flatten()
    hist = hist[hist.nonzero()]
    return scipy.stats.entropy(hist)

def dispersion_VCI(points, binsize=1):
    hist, bins = np.histogram(points['points'][:, 2],
                              bins=np.arange(np.min(points['points'][:, 2]), np.max(points['points'][:, 2]) + binsize, binsize),
                              density=True)
    hist = hist.flatten()
    hist = hist[hist.nonzero()]
    return -1 * np.sum(hist * np.log(hist)) / np.log(len(bins))

