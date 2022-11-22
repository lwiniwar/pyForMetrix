import warnings

import numpy as np
import scipy

from pyForMetrix.metricCalculators import MetricCalculator
from pyForMetrix.metrix import _rumple_index


class MCalc_EchoMetrics(MetricCalculator):
    name = "Echo metrics"

    def __init__(self):
        ...

    def get_names(self):
        return [
            "p_first_returns",
            "p_first_veg_returns"
        ]

    def __call__(self, points_in_poly: dict):
        points = points_in_poly['points']
        echo_number = points_in_poly['echo_number']
        classification = points_in_poly['classification']

        total_points = points.shape[0]

        outArray = np.full(((len(self.get_names())),), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0] = np.count_nonzero(echo_number == 1) / total_points
            outArray[1] = np.count_nonzero(np.logical_and(
                echo_number == 1, np.logical_and(classification >= 3, classification <= 5))
                # classes 3, 4, 5 are vegetation classes
            ) / total_points
        return outArray


class MCalc_HeightMetrics(MetricCalculator):
    name = "Height metrics"

    def __init__(self, percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])):
        self.p = np.array(percentiles)

    def get_names(self):
        return [
                   f"p{p}" for p in self.p] + \
               [
                   "h_mean",
               ]

    def __call__(self, points_in_poly: dict):
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())),), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0:len(self.p)] = np.percentile(points[:, 2], self.p)
            var_length_end = len(self.p)
            outArray[var_length_end] = np.mean(points[:, 2])
        return outArray


class MCalc_DensityMetrics(MetricCalculator):
    name = "Density metrics"

    def __init__(self, density_percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])):
        self.d = np.array(density_percentiles)

    def get_names(self):
        return [
            f"d{d}" for d in self.d]

    def __call__(self, points_in_poly: dict):
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())),), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_points = points.shape[0]
            max_height = np.max(points[:, 2])
            outArray[0:len(self.d)] = [np.count_nonzero(points[:, 2] > val) / total_points
                                       for val in (self.d / 100. * max_height)]
        return outArray


class MCalc_VarianceMetrics(MetricCalculator):
    name = "Variance metrics"

    def __init__(self):
        ...

    def get_names(self):
        return [
            "h_stddev",
            "h_absdev",
            "h_skew",
            "h_kurtosis",
            "h_entropy"
        ]

    def __call__(self, points_in_poly: dict):
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())),), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0] = np.std(points[:, 2], ddof=1)
            outArray[1] = np.mean(np.abs(np.mean(points[:, 2]) - points[:, 2]))
            outArray[2] = scipy.stats.skew(points[:, 2])
            outArray[3] = scipy.stats.kurtosis(points[:, 2])

            hist = np.histogramdd(points[:, 2], bins=20)[0]
            hist /= hist.sum()
            hist = hist.flatten()
            hist = hist[hist.nonzero()]
            outArray[4] = scipy.stats.entropy(hist)
        return outArray


class MCalc_CoverMetrics(MetricCalculator):
    name = "Variance metrics"

    def __init__(self):
        ...

    def get_names(self):
        return [
            "cover_cc2",
            "cover_ccmean",
            "cover_rumple",

        ]

    def __call__(self, points_in_poly: dict, rumple_pixel_size=1):
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())),), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            outArray[0] = np.count_nonzero(points[:, 2] > 2.0) / points.shape[0]
            outArray[1] = np.count_nonzero(points[:, 2] > np.mean(points[:, 2])) / points.shape[0]
            rumple = _rumple_index(points, rumple_pixel_size)
            outArray[2] = rumple
        return outArray


class MCalc_VisMetrics(MetricCalculator):
    """
    Calculate metrics for visualisation of the point cloud data

    :param points: np.array of shape (n, 3) with points in a single compute unit -- normalized height required
    :param progressbar: multiprocessing queue object to push updates to or None
    :return: np.array of shape (2, ) with metrics: <br />
        - Total number of points <br />
        - Max. height model (CHM) <br />
        - Number of unique strips at location <br />
    """
    name = "Visualisation only"

    def __init__(self):
        super(MCalc_VisMetrics, self).__init__()

    def get_names(self):
        return [
            'nPoints',
            'hmax',
            'uniqueStrips',
            'maxScanAngle'
        ]

    def __call__(self, points_in_poly:dict):
        points = points_in_poly['points']
        sar = points_in_poly['scan_angle_rank']
        outArray = np.full(((len(self.get_names())), ), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0] = points.shape[0]
            outArray[1] = np.max(points[:, 2])
            outArray[2] = len(np.unique(points_in_poly['pt_src_id']))
            outArray[3] = np.max(sar)
        return outArray