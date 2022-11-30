import warnings

import numpy as np
import scipy

from pyForMetrix.metricCalculators import MetricCalculator
from pyForMetrix.metricCalculators.lidRmetrics.rumple import rumple_index


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
            rumple = rumple_index(points_in_poly, rumple_pixel_size)
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

class MCalc_lidRmetrics_basic(MetricCalculator):
    name = "_basic"
    _names = ['n', 'zmax', 'zmin', 'zmean', 'zsd', 'zcv', 'zskew', 'zkurt']

    def __call__(self, points:dict):
        from pyForMetrix.metricCalculators.lidRmetrics import basic
        return np.array([
            basic.basic_n(points),
            basic.basic_zmax(points),
            basic.basic_zmin(points),
            basic.basic_zmean(points),
            basic.basic_zsd(points),
            basic.basic_zcv(points),
            basic.basic_zskew(points),
            basic.basic_zkurt(points)
            ])

class MCalc_lidRmetrics_percentiles(MetricCalculator):
    name = "_percentiles"
    def __init__(self, percentiles=np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                             55, 60, 65, 70, 75, 80, 85, 90, 95, 99])):
        self.percentiles = percentiles
        super(MCalc_lidRmetrics_percentiles, self).__init__()
    def get_names(self):
        return [f'zq{q:d}' for q in self.percentiles]

    def __call__(self, points:dict):
        from pyForMetrix.metricCalculators.lidRmetrics import percentiles
        return percentiles.percentiles_z(points, self.percentiles)

class MCalc_lidRmetrics_percabove(MetricCalculator):
    name = '_percabove'
    def __init__(self, p_above=np.array([2, 5])):
        self.p_above = p_above
        super(MCalc_lidRmetrics_percabove, self).__init__()
    def get_names(self):
        return ['pzabovemean'] + [f'pzabove{q}' for q in self.p_above]
    def __call__(self, points:dict):
        from pyForMetrix.metricCalculators.lidRmetrics import percabove
        outArray = np.full((len(self),), np.nan)
        outArray[0] = percabove.percabove_pzabovemean(points)
        outArray[1:] = [percabove.percabove_pzaboveX(points, X) for X in self.p_above]
        return outArray

class MCalc_lidRmetrics_dispersion(MetricCalculator):
    name = '_dispersion'
    _names = ['ziqr', 'zMADmean', 'zMADmedian', 'CRR', 'zentropy', 'VCI']
    def __call__(self, points:dict, binsize:float=1):
        from pyForMetrix.metricCalculators.lidRmetrics import dispersion
        return np.array([
            dispersion.dispersion_ziqr(points),
            dispersion.dispersion_zMADmean(points),
            dispersion.dispersion_zMADmedian(points),
            dispersion.dispersion_CRR(points),
            dispersion.dispersion_zentropy(points, binsize=binsize),
            dispersion.dispersion_VCI(points, binsize=binsize),
        ])

class MCalc_lidRmetrics_canopydensity(MetricCalculator):
    name = '_canopydensity'
    def __init__(self, num_groups:int=10):
        self.num_groups = num_groups
        super(MCalc_lidRmetrics_canopydensity, self).__init__()
    def get_names(self):
        return [f'zpcum{i:d}' for i in range(1, self.num_groups)]
    def __call__(self, points):
        from pyForMetrix.metricCalculators.lidRmetrics import canopydensity
        return canopydensity.canopydensity_zpcum(points, self.num_groups)



class MCalc_lidRmetrics_Lmoments(MetricCalculator):
    name = '_Lmoments'
    _names = ['L1', 'L2', 'L3', 'L4', 'Lskew', 'Lkurt', 'Lcoefvar']

    def __call__(self, points):
        from pyForMetrix.metricCalculators.lidRmetrics import Lmoments
        outArray = np.full((len(self),), np.nan)
        outArray[:4] = Lmoments.Lmoments_moments(points)
        outArray[4:] = Lmoments.Lmoments_coefficients(points)  # order: L_skew, L_kurt, L_CV
        return outArray
class MCalc_lidRmetrics_lad(MetricCalculator):
    """
    Following http://doi.org/10.1016/j.rse.2014.10.004
    """
    name = '_lad'
    _names = ['lad_max', 'lad_mean', 'lad_cv', 'lad_min']
    def __call__(self, points, dz=1):
        from pyForMetrix.metricCalculators.lidRmetrics import lad
        return lad.lad_lad(points, dz)
class MCalc_lidRmetrics_interval(MetricCalculator):
    name = "_interval"
    def __init__(self, intervals=np.array([0, 0.15, 2, 5, 10, 20, 30])):
        self.intervals = intervals
        super(MCalc_lidRmetrics_interval, self).__init__()
    def get_names(self):
        return [f'pz_below_{q}' for q in self.intervals]
    def __call__(self, points):
        from pyForMetrix.metricCalculators.lidRmetrics import interval
        return np.array([interval.interval_p_below(points, threshold=X) for X in self.intervals])
class MCalc_lidRmetrics_rumple(MetricCalculator):
    name = '_rumple'
    _names = ['rumple']
    def __call__(self, points, rumple_pixel_size=1):
        from pyForMetrix.metricCalculators.lidRmetrics import rumple
        return np.array([rumple.rumple_index(points, rumple_pixel_size)])

class MCalc_lidRmetrics_voxels(MetricCalculator):
    name = '_voxels'
    _names = ['vn', 'vFRall', 'vFRcanopy', 'vzrumple', 'vzsd', 'vzcv', 'OpenGapSpace', 'ClosedGapSpace', 'Euphotic', 'Oligophotic']

    def __call__(self, points, voxel_size=1):
        from pyForMetrix.metricCalculators.lidRmetrics import voxels
        outArray = np.full((len(self),), np.nan)
        XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex = voxels.create_voxelization(points, voxel_size)
        outArray[0] = voxels.voxels_vn(idxVoxelUnique)
        outArray[1] = voxels.voxels_vFRall(idxVoxelUnique)
        outArray[2] = voxels.voxels_vFRcanopy(idxVoxelUnique)
        outArray[3] = voxels.voxels_vzrumple(idxVoxelUnique, voxel_size)
        outArray[4] = voxels.voxels_vzsd(idxVoxelUnique, voxel_size)
        outArray[5] = voxels.voxels_vzcv(idxVoxelUnique, voxel_size)
        outArray[6:] = voxels.voxels_lefsky(idxVoxelUnique)
        return outArray

class MCalc_lidRmetrics_kde(MetricCalculator):
    name = '_kde'
    _names = ['kde_peaks_count', 'kde_peaks_elev', 'kde_peaks_value']

    def __call__(self, points, bw=2):
        from pyForMetrix.metricCalculators.lidRmetrics import kde
        count, elev, value = kde.kde_kde(points, bw)
        return np.array([count, np.mean(elev), np.mean(value)])

class MCalc_lidRmetrics_echo(MetricCalculator):
    name = '_echo'
    _names = ['pFirst', 'pIntermediate', 'pLast', 'pSingle', 'pMultiple']

    def __call__(self, points):
        from pyForMetrix.metricCalculators.lidRmetrics import echo
        return np.array([
            echo.echo_pFirst(points),
            echo.echo_pIntermediate(points),
            echo.echo_pLast(points),
            echo.echo_pSingle(points),
            echo.echo_pMultiple(points),
        ])
class MCalc_lidRmetrics_HOME(MetricCalculator):
    name = '_HOME'
    _names = ['HOME']
    def __call__(self, points):
        from pyForMetrix.metricCalculators.lidRmetrics import HOME
        return np.array([
            HOME.HOME_home(points),
        ])