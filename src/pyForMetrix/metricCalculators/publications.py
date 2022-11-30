import warnings

import numpy as np
import scipy

from pyForMetrix.utils.rasterizer import Rasterizer
from pyForMetrix.metricCalculators import MetricCalculator
from pyForMetrix.metricCalculators.lidRmetrics.rumple import rumple_index


class MCalc_White_et_al_2015(MetricCalculator):
    """
    Metric calculation class

    Calculate metrics following
    White et al. (2015):
    Comparing ALS and Image-Based Point Cloud Metrics and
    Modelled Forest Inventory Attributes in a Complex Coastal
    Forest Environment

    https://doi.org/10.3390/f6103704

    See Table 6  in the paper and :meth:`__call__` for more information.

    """
    name = "White et al. (2015)"

    @staticmethod
    def get_names():
        """
        List names of the generated metrics

        Returns:
            :class:`list`:
             list of strings with the metrics that will be generated.
        """
        return [
            "Hmean",
            "CoV",
            "Skewness",
            "Kurtosis",
            "P10",
            "P90",
            "CCmean",
            "Rumple"
        ]

    def __call__(self, points_in_poly, rumple_pixel_size=1):
        """
        Calculate the metrics

        Args:
            points_in_poly: :class:`dict` that contains a key `points`, pointing to a :class:`numpy.ndarray` of shape (n,3)
            rumple_pixel_size: pixel size used for rumple index calculation

        Returns:
            :class:`numpy.ndarray`:
            - Hmean
            - CoV
            - Skewness
            - Kurtosis
            - P10
            - P90
            - CCmean
            - Rumple

        """
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())), ), np.nan)
        points = points[points[:, 2] > 2]
        if points.shape[0] == 0:
            return outArray
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0] = np.mean(points[:, 2])
            outArray[1] = np.std(points[:, 2], ddof=1) / outArray[0]
            outArray[2] = scipy.stats.skew(points[:, 2])
            outArray[3] = scipy.stats.kurtosis(points[:, 2])
            outArray[4:6] = np.percentile(points[:, 2], [10, 90])
            outArray[6] = np.count_nonzero(points[:, 2] > outArray[0]) / points.shape[0]
            rumple = rumple_index(points_in_poly, rumple_pixel_size)
            outArray[7] = rumple

        return outArray


class MCalc_Hollaus_et_al_2009(MetricCalculator):
    """
    Metric calculation class

    Calculate metrics following
    Hollaus et al. (2009): Growing stock estimation for alpine forests in Austria: a robust lidar-based approach.

    https://doi.org/10.1139/X09-042

    See Table 3 in the paper and :meth:`__call__` for more information.

    Args:
        height_bins_upper: a :class:`numpy.ndarray` defining the upper limits for the height bin classes

    """
    name = "Hollaus et al. (2009)"

    def __init__(self, height_bins_upper=np.array([2, 5, 10, 15, 20, 25, 30, 35])):
        self.height_bins = np.array(height_bins_upper)


    def get_names(self):
        """
        List names of the generated metrics

        Returns:
            :class:`list`:
             list of strings with the metrics that will be generated. As the number of height bins
             is given by the user, the length of the list depends on the settings.

        """
        return [
            f"v_fe_i{h}" for h in range(len(self.height_bins))
        ]

    def __call__(self, points_in_poly, CHM_pixel_size=1):
        """
        Calculate the metrics

        Args:
            points_in_poly: :class:`dict` that contains a key `points`, pointing to a :class:`numpy.ndarray` of shape (n,3)
            CHM_pixel_size: the pixel size for the canopy height model calculation

        Returns:
            :class:`numpy.ndarray`:
            - relative count of first echoes in each canopy height class

        """
        points = points_in_poly['points']
        # take first echoes only
        points = points[points_in_poly['echo_number'] == 1]
        if len(points) == 0:
            return np.zeros((len(self), ))
        fe_counts = np.zeros((len(self), ))
        fe_sumCHM = np.zeros((len(self), ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # rasterize points
            ras = Rasterizer(points, raster_size=(CHM_pixel_size, CHM_pixel_size))
            XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex = ras.rasterize()
            for contents in XVoxelContains:
                points_in_cell = points[contents]
                cell_CHM = np.max(points_in_cell[:, 2])

                fe_counts[np.argmin([max(0, elem) for elem in cell_CHM - self.height_bins])] += len(contents)
                fe_sumCHM[np.argmin([max(0, elem) for elem in cell_CHM - self.height_bins])] += np.sum(points_in_cell[:, 2])
            total_fe = np.sum(fe_sumCHM)
            meanCHM = fe_sumCHM / fe_counts
            p_CHM = fe_counts / total_fe
            outArray = meanCHM * p_CHM
            outArray[np.isnan(outArray)] = 0
        return outArray


class MCalc_Xu_et_al_2019(MetricCalculator):
    """
    Metric calculation class

    Calculate metrics following Xu et al. (2019):
    Estimation of secondary forest parameters by integrating
    image and point cloud-based metrics acquired
    from unmanned aerial vehicle

    https://doi.org/10.1117/1.JRS.14.022204

    See Table 3 in the paper and :meth:`__call__` for more information.

    Args:
        percentiles: a :class:`numpy.ndarray` with values between 0 and 100, representing the percentiles to be calculated
        density_percentiles: a :class:`numpy.ndarray` with values between 0 and 100, representing the height percentiles for
                             which densities are calculated

    """

    name = "Xu et al. (2019)"

    def __init__(self, percentiles=np.array([10, 25, 30, 40, 60, 75, 85, 90]),
                 density_percentiles=np.array([10, 25, 30, 40, 60, 75, 85, 90])):
        self.p = np.array(percentiles)
        self.d = np.array(density_percentiles)


    def get_names(self):
        """
        List names of the generated metrics

        Returns:
            :class:`list`:
             list of strings with the metrics that will be generated. As the percentiles and density metrics
             can be of different lengths, the length of the list depends on the settings.

        """
        return [
            f"p{p}" for p in self.p] + \
            [f"d{d}" for d in self.d] + \
        [
            "h_mean",
            "h_max",
            "h_min",
            "h_cv",
        ]

    def __call__(self, points_in_poly):
        """
        Calculate the metrics

        Args:
            points_in_poly: points_in_poly: :class:`dict` that contains a key `points`, pointing to a :class:`numpy.ndarray` of shape (n,3)

        Returns:
            :class:`numpy.ndarray`:
            - Height percentiles (p10, p25, p30, p40, p60, p75, p85, p90)
            - Density metrics (d10, d25, d30, d40, d60, d75, d85, d90)

              (The proportion of points above the height percentiles, Shen et al., 2018: https://doi.org/10.3390/rs10111729)

            - Height variation metrics (h_mean, h_max, h_min, h_cv)

        """
        points = points_in_poly['points']
        outArray = np.full(((len(self.get_names())), ), np.nan)
        points = points[points[:, 2] > 2]
        if points.shape[0] < 1:
            return outArray
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0:len(self.p)] = np.percentile(points[:, 2], self.p)
            total_points = points.shape[0]
            var_length_end = len(self.p) + len(self.d)
            max_height = np.max(points[:, 2])
            outArray[len(self.p):var_length_end] = [np.count_nonzero(points[:, 2] > val) / total_points
                                                     for val in (self.d / 100. * max_height)]
            outArray[var_length_end] = np.mean(points[:, 2])
            outArray[var_length_end + 1] = max_height
            outArray[var_length_end + 2] = np.min(points[:, 2])
            outArray[var_length_end + 3] = np.std(points[:, 2], ddof=1) / outArray[var_length_end]
        return outArray



class MCalc_Woods_et_al_2009(MetricCalculator):
    """
    Metric calculation class

    Calculate metrics following Woods et al. (2009):
    Predicting forest stand variables from LiDAR data
    in the Great Lakes – St. Lawrence forest of Ontario

    https://doi.org/10.5558/tfc84827-6

    See Section "LIDAR based predictors"  in the paper and :meth:`__call__` for more information.

    Args:
        percentiles: a :class:`numpy.ndarray` with values between 0 and 100, representing the percentiles to be calculated
        density_percentiles: a :class:`numpy.ndarray` with values between 0 and 100, representing the height percentiles for
                             which densities are calculated
    """
    name = "Woods et al. (2009)"

    def __init__(self, percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                 density_percentiles=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])):
        self.p = np.array(percentiles)
        self.d = np.array(density_percentiles)


    def get_names(self):
        """
        List names of the generated metrics

        Returns:
            :class:`list`:
             list of strings with the metrics that will be generated. As the percentiles and density metrics
             can be of different lengths, the length of the list depends on the settings.
        """
        return [
            f"p{p}" for p in self.p] + \
            [f"d{d}" for d in self.d] + \
        [
            "h_mean",
            "h_stddev",
            "h_absdev",
            "h_skew",
            "h_kurtosis",
            "p_first_returns",
            "p_first_veg_returns"
        ]
    def __call__(self, points_in_poly:dict):
        """
        Calculate the metrics

        Args:
            points_in_poly: :class:`dict` that contains keys `points`, `echo_number` and `classification`.
                            Each of the keys points to a `numpy.ndarray` with n entries (3xn for `points`),

        Returns:
            :class:`numpy.ndarray`:
            - Statistical metrics (h_mean, h_stddev, h_absdev, h_skew, h_kurtosis)
            - Canopy height metrics (default: p10, p20, p30, p40, p50, p60, p70, p80, p90, p100)
            - Density metrics (default: d10, d20, d30, d40, d50, d60, d70, d80, d90)
              (The proportion of points above the height percentiles,
              Shen et al., 2018: https://doi.org/10.3390/rs10111729)
            - Fraction of first returns
            - Fraction of first returns in the vegetation class
        """
        points = points_in_poly['points']
        echo_number = points_in_poly['echo_number']
        classification = points_in_poly['classification']
        outArray = np.full(((len(self.get_names())), ), np.nan)
        # no height threshold used by Wood et al.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outArray[0:len(self.p)] = np.percentile(points[:, 2], self.p)
            total_points = points.shape[0]
            var_length_end = len(self.p) + len(self.d)
            max_height = np.max(points[:, 2])
            outArray[len(self.p):var_length_end] = [np.count_nonzero(points[:, 2] > val) / total_points
                                                     for val in (self.d / 100. * max_height)]
            outArray[var_length_end] = np.mean(points[:, 2])
            outArray[var_length_end + 1] = np.std(points[:, 2], ddof=1)
            outArray[var_length_end + 2] = np.mean(np.abs(outArray[var_length_end] - points[:, 2]))
            outArray[var_length_end + 3] = scipy.stats.skew(points[:, 2])
            outArray[var_length_end + 4] = scipy.stats.kurtosis(points[:, 2])
            outArray[var_length_end + 5] =  np.count_nonzero(echo_number == 1) / total_points
            outArray[var_length_end + 6] =  np.count_nonzero(np.logical_and(
                echo_number == 1, np.logical_and(classification >= 3, classification<=5))  # classes 3, 4, 5 are vegetation classes
            ) / total_points
        return outArray
