import numpy as np


def HOME_home(points, zmin=None):
    z = points['points'][:, 2]
    intensity = points['intensity']
    if zmin is not None:
        valid_idx = z >= zmin
        z = z[valid_idx]
        intensity = intensity[valid_idx]
    order = np.argsort(z)
    z = z[order]
    intensity = intensity[order]
    csum1 = np.cumsum(intensity).astype(float)
    csum2 = np.cumsum(intensity[::-1])[::-1].astype(float)
    diffc = np.diff(np.sign(csum1 - csum2))
    # import matplotlib.pyplot as plt
    # plt.plot(csum1, 'r-')
    # plt.plot(csum2, 'k-')
    # plt.plot(csum1-csum2, 'b--')
    # plt.show()
    loc = np.nonzero(diffc)
    return z[loc]





if __name__ == '__main__':
    import laspy

    f = laspy.read(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\4_plots_clipped\1_inv\PRF009.las")
    points = {'points': f.xyz, 'intensity': f.intensity}
    z = HOME_home(points)
    print(np.sum(points['intensity'][points['points'][:, 2] <= z]))
    print(np.sum(points['intensity'][points['points'][:, 2] >= z]))