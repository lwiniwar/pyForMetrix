from test_plot_metrics import ensure_netzkater_data, datadir


def test_group_metrics():
    from pyForMetrix.metricCalculators.types import MCalc_VisMetrics, MCalc_DensityMetrics, \
        MCalc_HeightMetrics, MCalc_EchoMetrics, MCalc_CoverMetrics, MCalc_VarianceMetrics
    from pyForMetrix.metrix import RasterMetrics
    from pyForMetrix.normalizer import normalize
    ensure_netzkater_data()
    import laspy
    data = laspy.read(datadir / 'las_623_5718_1_th_2014-2019.laz')
    points = {
        'points': data.xyz,
        'classification': data.classification,
        'echo_number': data.return_number,
        'scan_angle_rank': data.scan_angle_rank,
        'pt_src_id': data.point_source_id
    }
    normalize(points)
    rm = RasterMetrics(points, raster_size=25)
    mcs = [MCalc_EchoMetrics(), MCalc_DensityMetrics(), MCalc_CoverMetrics(),
           MCalc_VarianceMetrics(), MCalc_HeightMetrics(), MCalc_VisMetrics()]
    results = rm.calc_custom_metrics(mcs)
    assert results.shape == (40, 40, 35)
    assert abs(results.sel({'val':'p100'}).data.max() - 105.63799999) < 0.0001

    print(results)
