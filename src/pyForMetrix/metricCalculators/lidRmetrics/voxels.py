from functools import reduce, lru_cache
from operator import mul

import numpy as np
import scipy.stats

from pyForMetrix.utils.voxelizer import Voxelizer


def create_voxelization(points, voxel_size):
    vox = Voxelizer(points['points'], voxel_size=voxel_size)
    XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex = vox.voxelize()
    return XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex

#@lru_cache(100)
def create_histogram(z, voxel_size):
    return np.histogram(z, bins=np.arange(z.min(), z.max() + 2*voxel_size, voxel_size), density=True)[0]
#@lru_cache(100)
def voxels_vn(idxVoxelUnique):
    return idxVoxelUnique.shape[0]

def voxels_vFRall(idxVoxelUnique):
    spanAll = np.ptp(idxVoxelUnique, axis=0)
    vall = reduce(mul, spanAll)
    if vall == 0:
        return np.nan
    return voxels_vn(idxVoxelUnique) / vall

def voxels_vFRcanopy(idxVoxelUnique):
    idxVoxelUnique = idxVoxelUnique.astype(int)
    spanAll = np.ptp(idxVoxelUnique, axis=0)
    map2d = np.zeros((spanAll[0]+1, spanAll[1]+1))
    for (x,y,z) in idxVoxelUnique:
        map2d[x,y] = np.max([z, map2d[x,y]])
    vbelowcanopy = np.sum(map2d)
    return voxels_vn(idxVoxelUnique) / vbelowcanopy

def voxels_vzrumple(idxVoxelUnique, voxel_size):
    z = idxVoxelUnique[:, 2]
    hist = create_histogram(z, voxel_size)
    flength = np.sum(np.sqrt(np.square(hist) + voxel_size ** 2))
    fheight = len(hist) * voxel_size
    return flength/fheight

def voxels_vzsd(idxVoxelUnique, voxel_size):
    z = idxVoxelUnique[:, 2]
    hist = create_histogram(z, voxel_size)
    return np.std(hist)
def voxels_vzcv(idxVoxelUnique, voxel_size):
    z = idxVoxelUnique[:, 2]
    hist = create_histogram(z, voxel_size)
    return scipy.stats.variation(hist)

def voxels_lefsky(idxVoxelUnique):
    """

    Args:
        idxVoxelUnique:

    Returns: Percentages of:
        - Empty voxels above canopy
        - Empty voxels below canopy
        - Filled voxels in the top 65% of the canopy (Euphotic)
        - Filled voxels below the top 65% of the canopy (Oligophotic)
    """
    idxVoxelUnique = idxVoxelUnique.astype(int)
    spanAll = np.ptp(idxVoxelUnique, axis=0)
    map2d = np.zeros((spanAll[0]+1, spanAll[1]+1))
    for (x,y,z) in idxVoxelUnique:
        map2d[x,y] = np.max([z, map2d[x,y]])
    maxZ = np.max(map2d)
    counts = [0,0,0,0]
    for locx, locy in np.unique(idxVoxelUnique[:, :2], axis=0):
        voxels_at_xy = (idxVoxelUnique[:, 0] == locx) & (idxVoxelUnique[:, 1] == locy)
        counts[0] += maxZ - map2d[locx, locy]  # empty voxels above canopy
        counts[1] += map2d[locx, locy] - np.count_nonzero(  # empty voxels below canopy: height minus number of filled
            (voxels_at_xy))   # voxels at that location
        counts[2] += np.count_nonzero(voxels_at_xy &
                                      (idxVoxelUnique[:, 2] >= 0.65 * map2d[locx, locy]))
        counts[3] += np.count_nonzero(voxels_at_xy &
                                      (idxVoxelUnique[:, 2] < 0.65 * map2d[locx, locy]))
    # do we need to add full empty columns to the empty voxels above? I don't think so, as this would
    # include all the voxels outside of the mbr/polygon/...
    #  counts[0] += np.count_nonzero(map2d == 0) * maxZ
    counts = np.array(counts) / np.sum(counts) * 100.
    return counts

if __name__ == '__main__':
    import laspy
    f = laspy.read(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\4_plots_clipped\2_psp\PSP 005.las")
    points = {'points': f.xyz}
    XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex = create_voxelization(points, 1)
    voxels_vFRcanopy(idxVoxelUnique)