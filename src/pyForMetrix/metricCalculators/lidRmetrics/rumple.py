import numpy as np
import scipy

from pyForMetrix.utils.rasterizer import Rasterizer


def rumple_index(points, rumple_pixel_size):
    xyz = points['points']
    # rumple index
    ras = Rasterizer(xyz, raster_size=(rumple_pixel_size, rumple_pixel_size))
    CHM_x, CHM_y, CHM_z = ras.to_matrix(reducer=np.max)
    area_3d = 0
    CHM_xx, CHM_yy = np.meshgrid(CHM_x, CHM_y)
    raster_points = np.vstack([CHM_xx.flatten(), CHM_yy.flatten(), CHM_z.flatten()]).T
    raster_points = raster_points[np.logical_not(np.isnan(raster_points[:, 2])), :]
    if raster_points.shape[0] < 4:  # min. 4 points needed for convex hull
        return np.nan
    try:
        tri = scipy.spatial.Delaunay(raster_points[:, :2])
        for p1, p2, p3 in tri.simplices:
            a = raster_points[p2] - raster_points[p1]
            b = raster_points[p3] - raster_points[p1]
            # c = raster_points[p2] - raster_points[p3]
            tri_3d = np.linalg.norm(np.cross(a, b)) / 2
            area_3d += tri_3d
        hull_2d = scipy.spatial.ConvexHull(raster_points[:, :2])
    except Exception as e:
        # print(e)
        # print("Setting rumple index to nan and continuing...")
        return np.nan
    area_2d = hull_2d.volume  # volume in 2D is the area!
    rumple = area_3d / area_2d
    return rumple
