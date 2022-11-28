# pyForMetrix
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lwiniwar/pyForMetrix/HEAD?labpath=demo%2Fgetting_started.ipynb)
[![ReadTheDocs](https://readthedocs.org/projects/pyformetrix/badge/?version=latest)](https://pyformetrix.readthedocs.io/en/latest/)
[![FWF](https://img.shields.io/badge/Funding-FWF-green)](#acknowledgement)


`pyForMetrix` is a Python package to extract metrics commonly used in forestry from laser scanning/LiDAR data. Main functionalities include a plot-based and a pixel-based calculation, and handling of large datasets.

## Installation
`pyForMetrix` is packaged and delivered via PyPi, and can be installed using **pip**:

```bash
python -m pip install pyForMetrix
```

## Getting started 
 > Note: You can run this *Getting started* section on binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lwiniwar/pyForMetrix/HEAD?labpath=demo%2Fgetting_started.ipynb)

First, we need a point cloud dataset. You can use your own or download a sample dataset, e.g. from the City of Vancouver:
https://webtransfer.vancouver.ca/opendata/2018LiDAR/4830E_54560N.zip

Unzip this file after download to find a `.las`-File, which we will use in the following.

We need to read in the point cloud into a numpy array. Depending on the metrics we will derive later, 
different attributes also have to be loaded in. In this example, the 3D point cloud along with classification and
echo number information is required. For reading in the file, we use [`laspy`](https://laspy.readthedocs.io/).

```python
import numpy as np
import laspy

inFile = laspy.read(r"4830E_54560N.las")
coords = np.vstack([inFile.x,
                    inFile.y,
                    inFile.z]).transpose()
points = {
    'points': coords,
    'echo_number': inFile.return_number,
    'classification': inFile.classification
}
```

After importing the package `pyForMetrics`, we can create a `RasterMetrics` or a `PlotMetrics` object, depending on 
the application. Let's first work with `RasterMetrics`, which will calculate the set of metrics for each cell of a
raster overlaid on the point cloud data.

```python
from pyForMetrix.metrix import RasterMetrics
rm = RasterMetrics(points, raster_size=25)
```
The code above may take some time to run, as on the creation of the`RasterMetrics` object, the point cloud is rasterized
to the final cells. The runtime will increase with more points and a smaller raster size.

We then select which metrics we want to calculate. `pyForMetrix` comes with a number of predefined metrics, convieniently grouped in two collections: `publications`, where metrics from different publications in the literature are taken, and `types`, which groups metrics by their type. Later, we will see how to create your own metric calculators. For now, we will use the ones presented by Woods et al. (2009):

```python
from pyForMetrix.metricCalculators.publications import MCalc_Woods_et_al_2009
mc = MCalc_Woods_et_al_2009()
metrics = rm.calc_custom_metrics(metrics=mc)
```

With the last line, we created an [`xarray`](https://docs.xarray.dev/en/stable/)`.DataArray` object containing the metrics for each pixel:
```python
print(metrics)
```
```
<xarray.DataArray (y: 115, x: 83, val: 26)>
array([[[ 1.19169000e+03,  1.19212000e+03,  1.19236000e+03, ...,
         -1.26632802e+00,  7.51640760e-01,  0.00000000e+00],
        [ 1.19254700e+03,  1.19255400e+03,  1.19256100e+03, ...,
         -2.00000000e+00,  1.00000000e+00,  0.00000000e+00],
...
```

Using [`rioxarray`](https://corteva.github.io/rioxarray/stable/), we can save the values (here: the `p90` metric, i.e., the 90th height percentile) to a raster file:

```python
import rioxarray
metrics.sel(val='p90').rio.to_raster(f"p90.tif", "COG")
```

## More examples
### Multiple metric sets at once
Instead of passing a single `metricCalculator` class to `calc_custom_metrics`, you can call it with a list of `metricCalculator`s:
````python
from pyForMetrix.metricCalculators.types import MCalc_HeightMetrics, MCalc_DensityMetrics
heightMetrics = MCalc_HeightMetrics()
densityMetrics = MCalc_DensityMetrics()
metrics = rm.calc_custom_metrics(metrics=[heightMetrics, densityMetrics])
````
### Override percentiles, custom options
Some `metricCalculator`s can be customized, e.g. the `MCalc_HeightMetrics` accept an optional keyword `percentiles`, which
replaces the percentiles calculated by default:

````python
heightMetrics = MCalc_HeightMetrics(percentiles=np.array([15, 25, 50, 75, 85, 95, 99]))
````

Similarly, the cell size for the rumple index (e.g. in `MCalc_White_et_al_2015`) or the DSM in `MCalc_Hollaus_et_al_2009`
can be set - these variables are set as parameter to the `__call__` function. `calc_custom_metrics` accepts them as a (list of)
additional dictionaries with the settings:

````python
from pyForMetrix.metricCalculators.publications import MCalc_White_et_al_2015, MCalc_Hollaus_et_al_2009 
whiteMetrics = MCalc_White_et_al_2015()
metrics = rm.calc_custom_metrics(metrics=whiteMetrics, metric_options={'rumple_pixel_size': 0.2})
````
````python
hollausMetrics = MCalc_Hollaus_et_al_2009()
metrics = rm.calc_custom_metrics(metrics=[whiteMetrics, hollausMetrics], 
                                 metric_options=[
                                     {'rumple_pixel_size': 5},
                                     {'CHM_pixel_size': 7.5}
                                 ])
````
### Parallelize metric computation
On computers with multiple cores, processing can be sped up significantly by multiprocessing. To this end,
we provide a function `calc_custom_metrics_parallel` which takes similar arguments to `calc_custom_metrics`,
but runs on multiple cores. Note that the parallelization is carried out over the raster cells, i.e., the multiple
processes treat different subsets of the raster cells. As there is a certain overhead in starting the processes,
speedup is only expected if there is a large enough number of (a) valid raster cells and (b) metrics that are complex
to compute. The parameter `multiprocessing_point_threshold` checks the input point cloud and either spawns multiple processes
(in case the number of points is larger than the threshold) or passes the arguments on to `calc_custom_metrics`.

The other parameters are `n_chunks` (default: 16), which is the number of blocks the raster cells are divided into to be processed,
and `n_processes` (default: 4), which is the number of concurrent processes. A higher number of `n_chunks` uses less memory, but takes
longer due to the overhead.

On systems with sufficient memory (RAM > (number of processes) x (max. size of a tile)), it is generally better to parallelize over
input tiles rather than pixels.

### Plotwise metric extraction
You can find an example notebook for plotwise metric extraction [here](https://github.com/lwiniwar/pyForMetrix/blob/main/demo/plotwise_metrics.ipynb), or 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lwiniwar/pyForMetrix/HEAD?labpath=demo%2Fplotwise_metrics.ipynb) 

directly.

## Full / API documentation
The full documentation can be found at [readthedocs](https://pyformetrix.readthedocs.io/en/latest/).


## Dependencies
This package relies on the following packages (installed automatically when using pip). Thank you to all developers making this project possible!

- [`laxpy`](https://github.com/brycefrank/laxpy)
- [`numpy`](https://numpy.org/)
- [`scipy`](https://scipy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`tqdm`](https://tqdm.github.io/)
- [`xarray`](https://docs.xarray.dev/en/stable/)
- [`matplotlib`](https://matplotlib.org/)
- [`shapely`](https://shapely.readthedocs.io/en/stable/manual.html)

## Acknowledgement
This package has been developed in the course of the *UncertainTree* project, funded by the Austrian Science Fund ([FWF](https://www.fwf.ac.at/)) [Grant number J 4672-N].