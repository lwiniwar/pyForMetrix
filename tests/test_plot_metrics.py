import geopandas, pathlib
datadir = pathlib.Path(__file__).parent / '../demo'

def ensure_netzkater_data():
    import os, wget, zipfile
    if not os.path.exists(datadir / 'las_623_5718_1_th_2014-2019.laz'):
        if not os.path.exists(datadir / 'data_netzkater.zip'):
            print('Downloading file')
            wget.download(
                'https://geoportal.geoportal-th.de/hoehendaten/LAS/las_2014-2019/las_623_5718_1_th_2014-2019.zip',
                str((datadir / 'data_netzkater.zip').absolute()))
        print('Unzipping file')
        zipfile.ZipFile(str((datadir / 'data_netzkater.zip').absolute())).extractall(str(datadir.absolute()))


def test_paper_metrics():
    from pyForMetrix.metricCalculators.publications import MCalc_Hollaus_et_al_2009, MCalc_White_et_al_2015, \
        MCalc_Xu_et_al_2019, MCalc_Woods_et_al_2009
    from pyForMetrix.metrix import PlotMetrics
    ensure_netzkater_data()
    polys = geopandas.read_file(datadir / 'netzkater_polygons.gpkg')
    pm = PlotMetrics([datadir / 'las_623_5718_1_th_2014-2019.laz'], polys)
    mcs = [MCalc_Hollaus_et_al_2009(), MCalc_White_et_al_2015(), MCalc_Xu_et_al_2019(), MCalc_Woods_et_al_2009()]
    results = pm.calc_custom_metrics(mcs)
    assert results.shape == (7,62)
    print(results)

def test_lidRmetrics():
    from pyForMetrix.metricCalculators.types import \
        MCalc_lidRmetrics_lad, \
        MCalc_lidRmetrics_kde, \
        MCalc_lidRmetrics_dispersion, \
        MCalc_lidRmetrics_voxels, \
        MCalc_lidRmetrics_HOME, \
        MCalc_lidRmetrics_percabove, \
        MCalc_lidRmetrics_echo, \
        MCalc_lidRmetrics_basic, \
        MCalc_lidRmetrics_Lmoments, \
        MCalc_lidRmetrics_rumple, \
        MCalc_lidRmetrics_percentiles, \
        MCalc_lidRmetrics_interval, \
        MCalc_lidRmetrics_canopydensity

    from pyForMetrix.metrix import PlotMetrics
    ensure_netzkater_data()
    polys = geopandas.read_file(datadir / 'netzkater_polygons.gpkg')
    pm = PlotMetrics([datadir / 'las_623_5718_1_th_2014-2019.laz'], polys)
    mcs = [MCalc_lidRmetrics_lad(),
           MCalc_lidRmetrics_kde(),
           MCalc_lidRmetrics_dispersion(),
           MCalc_lidRmetrics_voxels(),
           MCalc_lidRmetrics_HOME(),
           MCalc_lidRmetrics_percabove(),
            MCalc_lidRmetrics_echo(),
           MCalc_lidRmetrics_basic(),
           MCalc_lidRmetrics_rumple(),
           MCalc_lidRmetrics_percentiles(),
           MCalc_lidRmetrics_interval(),
           MCalc_lidRmetrics_canopydensity()]
    results = pm.calc_custom_metrics(mcs)
    assert results.shape == (7,78)
    print(results)