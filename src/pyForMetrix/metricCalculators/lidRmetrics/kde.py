import numpy as np
import scipy.stats

def kde_kde(points, bw=2):
    z = points['points'][:, 2]
    try:
        kernel = scipy.stats.gaussian_kde(z, bw)
    except:
        # could fail because of singular matrices or too few entries
        return np.nan, np.nan, np.nan
    domain = np.arange(z.min(), z.max(), bw)
    estim = kernel(domain)
    peaks = np.argwhere(np.diff(np.sign(np.diff(estim))) < 0) +1
    # import matplotlib.pyplot as plt
    # plt.plot(domain, estim, 'b-', linewidth=0.4)
    # plt.plot(domain[peaks], estim[peaks], 'ro', markersize=2)
    # plt.show()
    n_peaks = len(peaks)
    elevs = estim[peaks][::-1]  # sort from top to bottom
    values = domain[peaks][::-1]
    return n_peaks, elevs, values


if __name__ == '__main__':
    import laspy

    f = laspy.read(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\4_plots_clipped\1_inv\PRF009.las")
    points = {'points': f.xyz}
    print(kde_kde(points, bw=0.05))