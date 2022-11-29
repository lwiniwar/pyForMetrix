import abc
import re
import os
import warnings
import time
import functools
import multiprocessing
import multiprocessing.shared_memory

from deprecated import deprecated

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import tqdm
import xarray

from shapely.geometry import Polygon
from matplotlib.path import Path as mplPath
from laxpy.tree import LAXTree
from laxpy.file import LAXParser
import laspy

from pyForMetrix.metricCalculators import MetricCalculator
from pyForMetrix.utils.rasterizer import Rasterizer

def parallel_raster_metrics_for_chunk(XVoxelCenter, XVoxelContains, inPoints,
                                      outArrayName, outArrayShape, outArrayType,
                                      raster_size, raster_min,
                                      perc, p_zabovex,
                                      progressbar):
    if progressbar is not None:
        progressbar.put((0, 1))
    shm = multiprocessing.shared_memory.SharedMemory(outArrayName)
    outArray = np.ndarray(outArrayShape, dtype=outArrayType, buffer=shm.buf)
    for xcenter, ycenter, contains in zip(XVoxelCenter[0], XVoxelCenter[1], XVoxelContains):
        cellY = int((xcenter - raster_size / 2 - raster_min[0]) / raster_size)  # note that xarray expects (y,x)
        cellX = int((ycenter - raster_size / 2 - raster_min[1]) / raster_size)
        points = inPoints[contains, :]
        cell_metrics = calc_standard_metrics(points, p_zabovex, perc, progressbar)
        outArray[cellX, cellY, :] = cell_metrics
    shm.close()
    if progressbar is not None:
        progressbar.put((0, -1))
def parallel_custom_raster_metrics_for_chunk(XVoxelCenter, XVoxelContains, inPoints,
                                      outArrayName, outArrayShape, outArrayType,
                                      raster_size, raster_min,
                                      progressbar, metric, metric_options):
    if progressbar is not None:
        progressbar.put((0, 1))
    shm = multiprocessing.shared_memory.SharedMemory(outArrayName)
    outArray = np.ndarray(outArrayShape, dtype=outArrayType, buffer=shm.buf)
    for xcenter, ycenter, contains in zip(XVoxelCenter[0], XVoxelCenter[1], XVoxelContains):
        cellY = int((xcenter - raster_size / 2 - raster_min[0]) / raster_size)  # note that xarray expects (y,x)
        cellX = int((ycenter - raster_size / 2 - raster_min[1]) / raster_size)
        points = {key: item[contains, ...] for key, item in inPoints.items()}

        out_metrics = []
        for mx, mo in zip(metric, metric_options):
            cell_metrics = mx(points, **mo)
            out_metrics.append(cell_metrics)
        outArray[cellX, cellY, :] = np.concatenate(out_metrics)
    shm.close()
    if progressbar is not None:
        progressbar.put((0, -1))


def calc_standard_metrics(points, p_zabovex, perc, progressbar):
    if progressbar is not None:
        progressbar.put((1, 0))
    outArray = np.full((8 + len(perc) + len(p_zabovex)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outArray[0] = points.shape[0]
        outArray[1] = np.prod(np.max(points[:, :2], axis=0) - np.min(points[:, :2], axis=0))
        outArray[2] = np.mean(points[:, 2])
        outArray[3] = np.std(points[:, 2])
        outArray[4] = scipy.stats.skew(points[:, 2])
        outArray[5] = scipy.stats.kurtosis(points[:, 2])

        hist = np.histogramdd(points[:, 2], bins=20)[0]
        hist /= hist.sum()
        hist = hist.flatten()
        hist = hist[hist.nonzero()]
        outArray[6] = scipy.stats.entropy(hist)

        outArray[7] = np.count_nonzero(points[:, 2] > outArray[2]) / points.shape[0]
        data_pos = 8 + len(perc)
        outArray[8:data_pos] = np.percentile(points[:, 2], perc)
        for x in p_zabovex:
            outArray[data_pos] = np.count_nonzero(points[:, 2] > (x + np.min(points[:, 2]))) / points.shape[0]
            data_pos += 1
    return outArray


def updatePbar(total, queue, maxProc, pbar_position):
    desc = "Computing raster metrics"
    pCount = 0
    pbar = tqdm.tqdm(total=total, ncols=150, desc=desc + " (%02d/%02d Process(es))" % (pCount, maxProc), position=pbar_position,
                     colour='#94f19b')
    pbar.update(0)
    while True:
        inc, process = queue.get()
        pbar.update(inc)
        if process != 0:
            pCount += process
            pbar.set_description(desc + " (%02d/%02d Process(es))" % (pCount, maxProc))


class Metrics(abc.ABC):
    def calc_metrics(self):
        raise NotImplementedError

    def calc_metrics_parallel(self):
        raise NotImplementedError


class RasterMetrics(Metrics):
    def __init__(self, points, raster_size, percentiles=np.arange(0, 101, 5), p_zabovex=None, silent=True, pbars=True,
                 raster_min=None, raster_max=None, origin=None):
        """
        Class to calculate metrics on a raster (cell) basis.

        Args:
            points: :class:`dict` containing keys 'points' and potentially other attributes, which are :numpy:ndarray s containing the points.
            raster_size: :class:`float` raster cell size used for calculation
            percentiles: deprecated
            p_zabovex: deprecated
            silent: deprecated
            pbars: :class:`bool` whether to show progress bars or not
            raster_min: :class:`numpy.ndarray` of shape `(2,)` with the minimum x/y coordinates for the raster (default: derive from point cloud)
            raster_max: :class:`numpy.ndarray` of shape `(2,)` with the maximum x/y coordinates for the raster (default: derive from point cloud)
            origin: :class:`numpy.ndarray` of shape `(2,)` with the origin x/y coordinates (pixel center) for the raster (default: same as `raster_min`)
        """
        self.pbars = pbars
        self.perc = percentiles
        self.p_zabovex = p_zabovex if p_zabovex is not None else []
        if not isinstance(self.p_zabovex, list):
            self.p_zabovex = [self.p_zabovex]
        self.raster_size = raster_size
        self.points = points
        coords = points['points']
        self.raster_min = np.min(coords[:, 0:2], axis=0) if raster_min is None else raster_min
        self.raster_max = np.max(coords[:, 0:2], axis=0) if raster_max is None else raster_max
        self.origin = origin if origin is not None else self.raster_min

        n = ((self.raster_min - self.origin) // self.raster_size).astype(int)  # find next integer multiple of origin
        self.raster_min = self.origin + n * self.raster_size

        self.raster_dims = (
                       int(np.ceil((self.raster_max[1] - self.raster_min[1]) / raster_size)),
                       int(np.ceil((self.raster_max[0] - self.raster_min[0]) / raster_size)))
        ts = time.time()
        # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
        r = Rasterizer(coords, (raster_size, raster_size))
        XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex = r.rasterize(origin=self.origin)
        if not silent:
            print(f"Rasterization complete in {(time.time() - ts):.2f} seconds.")
        self.XVoxelCenter = XVoxelCenter
        self.XVoxelContains = XVoxelContains

    def calc_custom_metrics(self, metrics: MetricCalculator, metric_options=None):
        """
        Calculates the given metrics on the point cloud this class was initialized on.

        Args:
            metrics: a single :class:`pyForMetrix.metricCalculators.MetricCalculator` instance or a :class:`list` of such classes
            metric_options: a :class:`list` of :class:`dict`s with options (kwargs) for each `MetricCalculator`, or None.

        Returns:
            An :class:`xarray.Dataset` containing the metric(s) in a raster grid
        """
        if not isinstance(metrics, list):
            metrics = [metrics]
        if metric_options is None:
            metric_options = [dict()] * len(metrics)

        num_feats = sum([len(m.get_names()) for m in metrics])
        data = np.full((self.raster_dims + (num_feats, )), np.nan, dtype=float)

        for xcenter, ycenter, contains in zip(self.XVoxelCenter[0], self.XVoxelCenter[1], self.XVoxelContains):
            cellY = int((xcenter - self.raster_size / 2 - self.raster_min[0]) / self.raster_size)  # note that xarray expects (y,x)
            cellX = int((ycenter - self.raster_size / 2 - self.raster_min[1]) / self.raster_size)
            points = {key: item[contains, ...] for key, item in self.points.items()}
            out_metrics = []

            for metric, metric_option in zip(metrics, metric_options):
                cell_metrics = metric(points, **metric_option)
                out_metrics.append(cell_metrics)
            data[cellX, cellY, :] = np.concatenate(out_metrics)
        return self.convert_to_custom_data_array(data, metrics)


    def calc_custom_metrics_parallel(self, metrics, n_chunks=16, n_processes=4, pbar_position=0,
                                     multiprocessing_point_threshold=10_000, metric_options=None):
        """
        Calculates the given metrics on the point cloud this class was initialized on, in parallel.
        Parallelization is achieved by spawning multiple processes for subsets of the raster cells. Note that
        it might be faster to parallelize over input datasets, if they are chunked.

        Args:
            metrics: see :func:`calc_custom_metrics`
            n_chunks: number of chunks to split the valid raster cells into (more chunks decrease memory usage)
            n_processes: number of processes to work on the chunks (more processes increase memory usage)
            pbar_position: deprecated
            multiprocessing_point_threshold: number of raster cells at which multiprocessing should be started. For
                relatively small datasets, the overhead of spawning extra processes outweights the benefit. Ideal setting
                depends on the features that are calculated
            metric_options: see :func:`calc_custom_metrics`

        Returns:
            An :class:`xarray.Dataset` containing the metric(s) in a raster grid

        """
        if not isinstance(metrics, list):
            metrics = [metrics]
        if metric_options is None:
            metric_options = [dict()] * len(metrics)

        # if there are actually rather few voxels (e.g., < 10,000), single thread is faster due to less overhead
        if len(self.XVoxelCenter[0]) < multiprocessing_point_threshold:
            return self.calc_custom_metrics(metrics=metrics, metric_options=metric_options)

        num_feats = sum([len(m.get_names()) for m in metrics])

        data = np.empty(self.raster_dims + (num_feats,), dtype=float)
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
        data_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        data_arr[:] = np.nan

        XVoxelCenterChunks = np.array_split(np.array(self.XVoxelCenter), n_chunks, axis=-1)
        XVoxelContainsChunks =  np.array_split(np.array(self.XVoxelContains, dtype=object), n_chunks)

        m = multiprocessing.Manager()
        if self.pbars:
            pbarQueue = m.Queue()
            pbarProc = multiprocessing.Process(target=updatePbar, args=(self.XVoxelCenter[0].shape[0], pbarQueue, n_processes, pbar_position))
            pbarProc.start()
        else:
            pbarQueue = None

        pool = multiprocessing.Pool(processes=n_processes)
        processing_function = functools.partial(parallel_custom_raster_metrics_for_chunk,
                                                inPoints=self.points,
                                                outArrayName=shm.name,
                                                outArrayShape=data_arr.shape,
                                                outArrayType=data_arr.dtype,
                                                raster_size=self.raster_size,
                                                raster_min=self.raster_min,
                                                progressbar=pbarQueue,
                                                metric = metrics,
                                                metric_options = metric_options)
        pool.starmap(processing_function, zip(XVoxelCenterChunks, XVoxelContainsChunks), chunksize=1)
        data[:] = data_arr[:]
        shm.close()
        shm.unlink()
        if self.pbars:
            pbarProc.kill()
        return self.convert_to_custom_data_array(data, metrics)

    @deprecated(version="0.0.5", reason="This function is being replaced by calc_custom_metrics")
    def calc_metrics(self,
                     progressbaropts=None,
                     pbar_position=0,
                     *args, **kwargs):
        if progressbaropts is None:
            progressbaropts = {'desc': 'Computing raster metrics (   Single Process)',
                               'ncols': 150,
                               'leave': False,
                               'colour': '#94f19b'}
        num_feats = len(self.perc) + 8 + len(self.p_zabovex)
        data = np.full(self.raster_dims + (num_feats, ), np.nan, dtype=float)

        for xcenter, ycenter, contains in zip(tqdm.tqdm(self.XVoxelCenter[0], position=pbar_position, **progressbaropts), self.XVoxelCenter[1], self.XVoxelContains):
            cellY = int((xcenter - self.raster_size / 2 - self.raster_min[0]) / self.raster_size)  # note that xarray expects (y,x)
            cellX = int((ycenter - self.raster_size / 2 - self.raster_min[1]) / self.raster_size)
            points = self.points["points"][contains, :]
            cell_metrics = calc_standard_metrics(points, self.p_zabovex, self.perc, None)
            data[cellX, cellY, :] = cell_metrics
        return self.convert_to_data_array(data)

    def convert_to_custom_data_array(self, data, metrics):
        return xarray.DataArray(data, dims=('y', 'x', 'val'),
                         coords={'y': np.arange(self.raster_min[1], self.raster_max[1], self.raster_size) + self.raster_size/2,
                         # coords={'y': np.arange(self.raster_min[1], self.raster_max[1], self.raster_dims[1]) + self.raster_size/2,
                                 'x': np.arange(self.raster_min[0], self.raster_max[0], self.raster_size) + self.raster_size/2,
                                 # 'x': np.linspace(self.raster_min[0], self.raster_max[0], self.raster_dims[0]) + self.raster_size/2,
                                 'val': np.concatenate([m.get_names() for m in metrics])
                                 })

    @deprecated(version="0.0.5", reason="This function is being replaced by convert_to_custom_data_array")
    def convert_to_data_array(self, data):
        return xarray.DataArray(data, dims=('y', 'x', 'val'),
                         coords={'y': np.arange(self.raster_min[1], self.raster_max[1], self.raster_size) + self.raster_size/2,
                                 'x': np.arange(self.raster_min[0], self.raster_max[0], self.raster_size) + self.raster_size/2,
                                 'val': [
                                            'n',
                                            'area',
                                            'meanZ',
                                            'stdZ',
                                            'skewZ',
                                            'kurtZ',
                                            'entropyZ',
                                            'nAboveMean'
                                        ] +
                                        [f"perc{p}Z" for p in self.perc] +
                                        [f"pAboveX{x}Z" for x in self.p_zabovex]
                                 })

    @deprecated(version="0.0.5", reason="This function is being replaced by calc_custom_metrics_parallel")
    def calc_metrics_parallel(self, n_chunks=16, n_processes=4, pbar_position=0, *args, **kwargs):
        # if there are actually rather few voxels (e.g., < 10,000), single thread is faster due to less overhead
        if len(self.XVoxelCenter[0]) < 10_000:
            return self.calc_metrics(pbar_position=pbar_position)

        num_feats = len(self.perc) + 8 + len(self.p_zabovex)

        data = np.empty(self.raster_dims + (num_feats,), dtype=float)
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
        data_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        data_arr[:] = np.nan

        XVoxelCenterChunks = np.array_split(np.array(self.XVoxelCenter), n_chunks, axis=-1)
        XVoxelContainsChunks =  np.array_split(np.array(self.XVoxelContains, dtype=object), n_chunks)

        m = multiprocessing.Manager()
        if self.pbars:
            pbarQueue = m.Queue()
            pbarProc = multiprocessing.Process(target=updatePbar, args=(self.XVoxelCenter[0].shape[0], pbarQueue, n_processes, pbar_position))
            pbarProc.start()
        else:
            pbarQueue = None

        pool = multiprocessing.Pool(processes=n_processes)
        processing_function = functools.partial(parallel_raster_metrics_for_chunk,
                                                inPoints=self.coords,
                                                outArrayName=shm.name,
                                                outArrayShape=data_arr.shape,
                                                outArrayType=data_arr.dtype,
                                                raster_size=self.raster_size,
                                                raster_min=self.raster_min,
                                                perc=self.perc,
                                                p_zabovex=self.p_zabovex,
                                                progressbar=pbarQueue)
        pool.starmap(processing_function, zip(XVoxelCenterChunks, XVoxelContainsChunks), chunksize=1)
        data[:] = data_arr[:]
        shm.close()
        shm.unlink()
        if self.pbars:
            pbarProc.kill()
        return self.convert_to_data_array(data)


class PlotMetrics(Metrics):
    def __init__(self, lasfiles, plot_polygons, silent=True, pbars=True):
        """
        Class to calculate metrics on a plot (polygon) basis

        Args:
            lasfiles: :class:`list` of input las-Files to consider. Note that the scanning (finding the points inside
                the plots) can be sped up siginificantly by providing `.lax`-Files, which can be generated e.g. using
                lasindex, part of the LASTools (https://rapidlasso.com/lastools/, proprietory software with free/open
                components).
            plot_polygons: :class:`geopandas.GeoDataFrame` array containing the geometries (polygons) of interest
            silent: :class:`boolean` whether to print output or not
            pbars: :class:`boolean` whether to display progress bars or not
        """
        self.lasfiles = lasfiles
        self.plot_polygons = plot_polygons
        self.silent = silent
        self.pbars = pbars

        # find points that are in the polygons
        self.points = [
            {
                'points': np.empty((0, 3), dtype=float),
                'echo_number': np.empty((0, ), dtype=int),
                'number_of_echoes': np.empty((0, ), dtype=int),
                'intensity': np.empty((0, ), dtype=float),
                'classification': np.empty((0, ), dtype=int),
                'pt_src_id': np.empty((0, ), dtype=int),
                'scan_angle_rank': np.empty((0, ), dtype=int),
            } for i in range(len(plot_polygons))
        ]
        if not isinstance(lasfiles, list):
            lasfiles = [lasfiles]

        for lasfile in tqdm.tqdm(self.lasfiles, ncols=150, desc='Scanning input files to find polygon plots'):
            laxfile = re.sub(r'^(.*).la[sz]$', r'\1.lax', str(lasfile))
            inFile = laspy.read(lasfile)
            if not os.path.exists(laxfile):
                print(f"File {lasfile} does not have a corresponding .lax index file. Expect much slower performance.")
                print(f"Run `lasindex -i {lasfile}` to create an index file (requires LAStools installation)")
            else:
                parser = LAXParser(laxfile)
                tree = LAXTree(parser)
            for q_id, q in plot_polygons.iterrows():
                q_polygon = q.geometry
                candidate_indices = []

                if not os.path.exists(laxfile):
                    candidate_indices = [np.arange(0, inFile.header.point_count)]  # brute force through all points
                else:
                    minx, maxx, miny, maxy = parser.bbox
                    bbox = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
                    if not q_polygon.intersects(bbox):
                        continue
                    for cell_index, polygon in tree.cell_polygons.items():  # use quadtree for preselection
                        if q_polygon.intersects(polygon):
                            candidate_indices.append(parser.create_point_indices(cell_index))

                if len(candidate_indices) > 0:  # brute force the rest
                    candidate_indices = np.unique(np.concatenate(candidate_indices))
                    p = mplPath(list(q_polygon.exterior.coords))
                    candidate_points = np.vstack((inFile.x[candidate_indices], inFile.y[candidate_indices])).T
                    is_inside = p.contains_points(candidate_points)
                    points_sel = np.argwhere(is_inside).flatten()
                    final_selection = candidate_indices[points_sel] #[::nth_point_subsample]
                    self.points[q_id]['points'] = np.concatenate((self.points[q_id]['points'], inFile.xyz[final_selection, :]), axis=0)
                    self.points[q_id]['echo_number'] = np.concatenate((self.points[q_id]['echo_number'], inFile.return_number[final_selection]), axis=0)
                    self.points[q_id]['number_of_echoes'] = np.concatenate((self.points[q_id]['number_of_echoes'], inFile.number_of_returns[final_selection]), axis=0)
                    self.points[q_id]['intensity'] = np.concatenate((self.points[q_id]['intensity'], inFile.intensity[final_selection]), axis=0)
                    self.points[q_id]['classification'] = np.concatenate((self.points[q_id]['classification'], inFile.classification[final_selection]), axis=0)
                    self.points[q_id]['pt_src_id'] = np.concatenate((self.points[q_id]['pt_src_id'], inFile.pt_src_id[final_selection]), axis=0)
                    self.points[q_id]['scan_angle_rank'] = np.concatenate((self.points[q_id]['scan_angle_rank'],
                                                                           inFile.scan_angle_rank[final_selection] if hasattr(inFile, 'scan_angle_rank') else inFile.scan_angle[final_selection]
                                                                           ), axis=0)

    def calc_custom_metrics(self, metrics: MetricCalculator, metric_options=None):
        """
        Calculates given metrics for points contained in the polygons given during construction of this class.

        Args:
            metrics: a single :class:`pyForMetrix.metricCalculators.MetricCalculator` instance or a :class:`list` of such classes
            metric_options: a :class:`list` of :class:`dict`s with options (kwargs) for each `MetricCalculator`, or None.

        Returns:
            a :class:`pandas.DataFrame` containing the metrics for each polygon in the input.

        """
        if metric_options is None:
            metric_options = dict()
        out_metrics = np.full((len(self.plot_polygons), sum(map(lambda x: len(x.get_names()), metrics))), np.nan)
        # plot_names = []
        if not self.silent:
            print('Calculating features for plot polygons...', end='')
        for q_id, q in tqdm.tqdm(self.plot_polygons.iterrows(), f"Calculating metrics", total=len(self.plot_polygons)):
            points_in_poly = self.points[q_id]
            if len(points_in_poly['points']) > 0:
                out_metrics[q_id] = np.concatenate(list(map(lambda x: x(points_in_poly, **metric_options), metrics)))
        out_data = pd.DataFrame(out_metrics,  # index=plot_names,
                                columns=
                                np.concatenate(list(map(lambda x: x.get_names(), metrics)))
                                )

        if not self.silent:
            print(' [done]')
        return out_data

    def calc_custom_metrics_stripwise(self, metrics: MetricCalculator, metric_options=None):
        if metric_options is None:
            metric_options = dict()
        out_metrics = []
        meta_metrics = []
        # plot_names = []
        if not self.silent:
            print('Calculating features for plot polygons...', end='')
        for q_id, q in tqdm.tqdm(self.plot_polygons.iterrows(), f"Calculating metrics", total=len(self.plot_polygons)):
            points_in_poly = self.points[q_id]
            unique_strips = np.unique(points_in_poly['pt_src_id'])
            for strip in unique_strips:
                points_in_poly_and_strip = {k: v[points_in_poly['pt_src_id'] == strip] for k, v in points_in_poly.items()}
                if points_in_poly_and_strip['points'].shape[0] > 3:
                    areaPoly = q.geometry.area
                    areaPc = scipy.spatial.ConvexHull(points_in_poly_and_strip['points'][:, :2]).volume
                    out_metrics.append(np.concatenate(list(map(lambda x: x(points_in_poly_and_strip, **metric_options), metrics))))
                    meta_metrics.append(np.array([q_id, areaPc, areaPoly, len(points_in_poly_and_strip['points']), strip,
                                                  np.min(points_in_poly_and_strip['scan_angle_rank']),
                                                  np.max(points_in_poly_and_strip['scan_angle_rank']),
                                                  np.mean(points_in_poly_and_strip['scan_angle_rank']),
                                                  ]))

        out_data = pd.DataFrame(out_metrics,  # index=plot_names,
                                columns=
                                np.concatenate(list(map(lambda x: x.get_names(), metrics)))
                                )
        out_meta = pd.DataFrame(meta_metrics, columns=['plot_id', 'areaPC', 'areaPoly', 'numPts', 'stripid', 'minSA', 'maxSA', 'meanSA'])

        if not self.silent:
            print(' [done]')
        return out_data, out_meta

    @deprecated(version="0.0.5", reason="This function is being replaced by calc_custom_metrics")

    def calc_metrics(self):
        out_metrics = np.full((len(self.plot_polygons), 8 + len(self.perc) + len(self.p_zabovex)), np.nan)
        # plot_names = []
        if not self.silent:
            print('Calculating features for plot polygons...', end='')
        for q_id, q in self.plot_polygons.iterrows():
            # plot_names.append(q.PLOT)
            if self.coords[q_id].shape[0] > 0:
                out_metrics[q_id] = calc_standard_metrics(self.coords[q_id], self.p_zabovex, self.perc, None)
        out_data = pd.DataFrame(out_metrics, # index=plot_names,
                                columns=
                                [
                                    'n',
                                    'area',
                                    'meanZ',
                                    'stdZ',
                                    'skewZ',
                                    'kurtZ',
                                    'entropyZ',
                                    'nAboveMean'
                                ] +
                                [f"perc{p}Z" for p in self.perc] +
                                [f"pAboveX{x}Z" for x in self.p_zabovex]
                                )

        if not self.silent:
            print(' [done]')
        return out_data

if __name__ == '__main__':
    if True:
        import laspy
        import cmcrameri.cm as cmc
        ts = time.time()
        # inFile = laspy.read(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\2_tiled\merged.laz")
        inFile = laspy.read(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\3_tiled_norm\300000_5094000.laz")
        coords = {
            'points': inFile.xyz,
            'echo_number': inFile.return_number,
            'number_of_echos': inFile.number_of_returns,
            'intensity': inFile.intensity,
            'classification': inFile.classification,

        }
        print(f"Data read in {time.time()-ts:.2f} seconds.")

        rm = RasterMetrics(coords, raster_size=20, percentiles=[25, 50, 75, 95, 99], p_zabovex=2)
        mc = MCalc_Woods_et_al_2009()
        ts = time.time()
        # metrix = rm.calc_metrics_parallel(n_chunks=10, n_processes=10)
        metrics = rm.calc_custom_metrics_parallel(metrics=mc, n_chunks=10, n_processes=10)
        print(f"Metrics computed in {time.time()-ts:.2f} seconds.")
        ts = time.time()
        metrix2 = rm.calc_custom_metrics(metrics=mc)
        print(f"Metrics computed in {time.time()-ts:.2f} seconds.")
        print(metrics)
        import matplotlib.pyplot as plt
        qminmax = metrics.quantile([0.05, 0.95], dim=['x', 'y'])
        for metric in metrics['val'].values:
            plt.title(metric)
            plt.imshow(metrics.sel(val=metric),
                       vmin=qminmax.sel(val=metric)[0],
                       vmax=qminmax.sel(val=metric)[1],
                       cmap=cmc.bamako)
            plt.colorbar()
            plt.show()
            plt.close()

    if False:
        import geopandas as gpd
        from pathlib import Path

        lasdir = Path(r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\Harmonized\2016_ALS\2_tiled")
        plots = gpd.GeoDataFrame.from_file(
            r"C:\Users\Lukas\Documents\Data\PetawawaHarmonized\prf_forest_sample_plots_2014\Polygons\AFRITPRF_Polys_Sept26_14.shp")
        pm = PlotMetrics(list(lasdir.glob('*00.laz')), plots)
        metrix = pm.calc_metrics()