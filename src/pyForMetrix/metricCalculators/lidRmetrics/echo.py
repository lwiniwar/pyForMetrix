import numpy as np

from pyForMetrix.metricCalculators.lidRmetrics.basic import basic_n

def echo_pFirst(points):
    n = basic_n(points)
    first = np.count_nonzero(points['echo_number'] == 1)
    return first/n
def echo_pIntermediate(points):
    n = basic_n(points)
    intermediate = np.count_nonzero((points['echo_number'] > 1) & (points['echo_number'] != points['number_of_echoes']))
    return intermediate/n

def echo_pLast(points):
    n = basic_n(points)
    last = np.count_nonzero((points['echo_number'] == points['number_of_echoes']) & points['echo_number'] > 1)  # excluding single echoes
    return last/n

def echo_pSingle(points):
    n = basic_n(points)
    single = np.count_nonzero(points['number_of_echoes'] == 1)
    return single/n
def echo_pMultiple(points):
    n = basic_n(points)
    multiple = np.count_nonzero(points['number_of_echoes'] > 1)
    return multiple/n