from __future__ import print_function, division
import numpy as np


class Voxelizer:
    def __init__(self, data, voxel_size=(1, 1, 1), method="random"):
        self.data = data
        if type(voxel_size) is not tuple:
            voxel_size = (voxel_size, voxel_size, voxel_size)
        self.voxel_size = voxel_size
        self.method = method

    def voxelize(self, origin=None):
        """
        Function to voxelize point cloud data
        Adapted from Glira (https://github.com/pglira/Point_cloud_tools_for_Matlab/
        blob/master/classes/4pointCloud/uniformSampling.m)

        :return:
        """
        # No.of points
        noPoi = self.data.shape[0]

        if origin is None:
            # Find voxel centers
            # Point with smallest coordinates
            minPoi = np.min(self.data, axis=0)

            # Rounded local origin for voxel structure
            # (voxels of different pcs have coincident voxel centers if mod(100, voxelSize) == 0)
            # localOrigin = np.floor(minPoi / 100) * 100
            localOrigin = np.floor(minPoi / 1) * 1
        else:
            localOrigin = origin

        # Find 3 - dimensional indices of voxels in which points are lying
        idxVoxel = np.array([np.floor((self.data[:, 0] - localOrigin[0]) / self.voxel_size[0]),
                             np.floor((self.data[:, 1] - localOrigin[1]) / self.voxel_size[1]),
                             np.floor((self.data[:, 2] - localOrigin[2]) / self.voxel_size[2])]).T

        # Remove multiple voxels
        idxVoxelUnique, ic = np.unique(idxVoxel, axis=0,
                                       return_inverse=True)  # ic contains "voxel index" for each point

        # Coordinates of voxel centers
        XVoxelCenter = [localOrigin[0] + self.voxel_size[0] / 2 + idxVoxelUnique[:, 0] * self.voxel_size[0],
                        localOrigin[1] + self.voxel_size[1] / 2 + idxVoxelUnique[:, 1] * self.voxel_size[1],
                        localOrigin[2] + self.voxel_size[2] / 2 + idxVoxelUnique[:, 2] * self.voxel_size[2]]

        # No.of voxel(equal to no.of selected points)
        noVoxel = len(XVoxelCenter[0])

        # Prepare list for every output voxel
        XVoxelContains = [[] for i in range(noVoxel)]
        XClosestIndex = np.full((noVoxel,), np.nan, dtype=np.int)

        # Select points nearest to voxel centers - --------------------------------------

        # Sort indices and points( in order to find points inside of voxels very fast in the next loop)
        idxSort = np.argsort(ic)
        ic = ic[idxSort]

        data_sorted = self.data[idxSort, :]
        idxJump, = np.nonzero(np.diff(ic))
        idxJump += 1

        # Example (3 voxel)
        # ic = [1 1 1 2 2 2 3]';
        # diff(ic) = [0 0 1 0 0 1]';
        # idxJump = [3     6]';
        #
        # idxInVoxel = [1 2 3]; for voxel 1
        # idxInVoxel = [4 5 6]; for voxel 2
        # idxInVoxel = [7];     for voxel 3

        for i in range(noVoxel):
            # Find indices of points inside of voxel(very, very fast this way)
            if i == 0:
                if i == noVoxel - 1:
                    idxInVoxel = slice(0, noPoi)
                else:
                    idxInVoxel = slice(0, idxJump[i])
            elif i == noVoxel - 1:
                idxInVoxel = slice(idxJump[i - 1], noPoi)
            else:
                idxInVoxel = slice(idxJump[i - 1], idxJump[i])

            # Fill voxel information
            XVoxelContains[i] = np.array(idxSort[idxInVoxel], dtype=int)

            # Get point closest to voxel center
            if self.method == "closest":
                distsSq = ((data_sorted[idxInVoxel, 0] - XVoxelCenter[0][i]) ** 2 +
                           (data_sorted[idxInVoxel, 1] - XVoxelCenter[1][i]) ** 2 +
                           (data_sorted[idxInVoxel, 2] - XVoxelCenter[2][i]) ** 2)
                closestIdxInVoxel = np.argmin(distsSq)
                XClosestIndex[i] = idxSort[idxInVoxel.start + closestIdxInVoxel]
            elif self.method == "random":
                XClosestIndex[i] = np.random.choice(XVoxelContains[i])

        return XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex