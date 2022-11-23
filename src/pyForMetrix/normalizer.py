import enum

import numpy as np
import tqdm
from scipy.spatial import cKDTree

from pyForMetrix.utils.rasterizer import Rasterizer

class NormalizeMode(enum.Enum):
    CylinderMode = 0
    CellMode = 1

def normalize(points, distance=3, mode=NormalizeMode.CellMode, percentile=5, show_progress=True, add_as_entry=False):
    """
    Function to normalize a point cloud.
    Args:
        points: :class:dict with a key 'points', which contains a n x 3 :class:numpy.ndarray with point coordinates
        distance: :class:int, the search radius or the cell size (depending on the `mode`)
        mode: :class:NormalizeMode
        percentile: :class:float, which percentile to use as 'ground' (default: 5, 0<=value<=100)
        show_progress: :class:bool, whether to print progress or not.
        add_as_entry: :class:bool, whether to add the normalized height as an entry to the input dictionary (key `nZ`)
        or to overwrite z coordinates (default)

    Returns: a pointer to the input `points` :class:dict, with an added key `nZ` containing the normalized height
    (in case add_as_entry is set to `True`), or with the z-Value overwritten (default).

    """
    nZ = np.copy(points['points'][:, 2])

    if mode == NormalizeMode.CylinderMode:
        tree = cKDTree(points['points'][:, 0:2])  # build kD-Tree on x/z coordinates
        if show_progress:
            print("Building kD-Tree for point normalization...")
        neighbours = tree.query_ball_tree(tree, r=distance)
        # get percentile for normalization
        it = tqdm.tqdm(enumerate(neighbours), desc="Normalizing LiDAR points") if show_progress else enumerate(neighbours)
        for nIdx, neighbour in it:
            perc = np.percentile(neighbour, percentile)
            nZ[nIdx] -= perc
    elif mode == NormalizeMode.CellMode:
        ras = Rasterizer(points['points'][:, 0:2], raster_size=distance)
        if show_progress:
            print("Rasterizing for point normalization...")
        _, XVoxelContains, _, _ = ras.rasterize()
        it = tqdm.tqdm(XVoxelContains, desc="Normalizing LiDAR points") if show_progress else XVoxelContains
        for contains in it:
        # get percentile for normalization
            perc = np.percentile(points['points'][contains, 2], percentile)
            nZ[contains] -= perc
    else:
        raise NotImplementedError(f"Unknown NormalizeMode: '{mode}'. "
                                  f"Supported modes: {', '.join([mode.name for mode in NormalizeMode])}")

    points['points'][:,2] = nZ
    return points


if __name__ == '__main__':
    from pyForMetrix.metricCalculators import MetricCalculator
    import warnings

    class CHMmetric(MetricCalculator):
        name = "chm"
        def get_names(self):
            return [
                'chm'
            ]

        def __call__(self, points_in_poly: dict):
            points = points_in_poly['points']
            outArray = np.full((len(self),), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outArray[0] = np.max(points[:, 2])
            return outArray


    import laspy
    file = laspy.read(r"C:\Users\Lukas\Documents\Projects\pyForMetrix\demo\las_623_5718_1_th_2014-2019.laz")
    points = {
        'points': file.xyz
    }
    points = normalize(points, percentile=1)
    print(np.ptp(points['points'][:, 2]))
    print(np.ptp(points['nZ']))

    origZ = points['points'][:, 2].copy()

    from pyForMetrix.metrix import RasterMetrics
    for z in (origZ, points['points'][:, 2]):
        pts = points.copy()
        pts['points'][:, 2] = z
        rm = RasterMetrics(pts, raster_size=5)
        mc = CHMmetric()
        rasters = rm.calc_custom_metrics(mc)

        from matplotlib import pyplot as plt
        plt.imshow(rasters.sel(val='chm'))
        plt.show()