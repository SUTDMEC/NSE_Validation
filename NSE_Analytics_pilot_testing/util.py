"""
Helper functions and data types.
"""

from collections import namedtuple
from itertools import tee, izip
import math
import pandas as pd
from operator import eq
import numpy as np

# mean earth radius in kilometers
# https://en.wikipedia.org/wiki/Earth_radius
earth_radius = 6371.0



def great_circle_dist(a, b, unit="kilometers"):
    """
    compute great circle distance between two latitude/longitude coordinate pairs.
    Returns great cirlce distance in kilometers (default) or meters.
    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    lat1, lon1 = a
    lat2, lon2 = b
    if (lat1==92) or (lat2==92):
        return -1 # invalid location gives invalid distance
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
            math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2.0 * math.asin(math.sqrt(a))
    dist_km = earth_radius * c
    if unit == "kilometers":
        return dist_km
    elif unit == "meters":
        return dist_km * 1000
    else:
        raise ValueError("Unknown unit: %s" % unit)

    
def dist_to_radians(x):
    return (x / earth_radius) * (180.0 / math.pi)


def sliding_window(iterable, size):
    """ Yield moving windows of size 'size' over the iterable object.
    
    Example:
    >>> for each in window(xrange(6), 3):
    >>>     print list(each)
    [0, 1, 1]
    [1, 2, 2]
    [2, 3, 3]
    [3, 4, 4]

    """
    iters = tee(iterable, size)
    for i in xrange(1, size):
        for each in iters[i:]:
            next(each, None)
    return izip(*iters)
    
                    
def print_full_dataframe(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')


def chunks(iterable, include_values=False, equal=eq):
    """Given an iterable, yield tuples of (start, end) indices for chunks
    with equal items. If inlcude_values is True, each tuple will have
    the value of that chunk as a third element.

    Example:
    >>> list(chunks([1, 1, 1, 2, 2, 1]))
    [(0, 3), (3, 5), (5, 6)]

    """
    idx = None
    start_idx = 0
    for idx, item in enumerate(iterable):
        if idx == 0:
            previous = item
            continue
        if not equal(item, previous):
            if include_values:
                yield (start_idx, idx, previous)
            else:
                yield (start_idx, idx)
            start_idx = idx
            previous = item
    if idx is not None:
        if include_values:
            yield (start_idx, idx+1, previous)
        else:
            yield (start_idx, idx+1)


def aveWithNan( data_vec ):
    #  function used to calculate the average of a vector
    #  able to deal with vector which has NaN points.

    idx_not_nan = np.where(~np.isnan(data_vec))[0];
    if len(idx_not_nan) == 0:
        # all nan in data_to_cal
        mean_data = np.nan
    else:
        mean_data = np.mean(data_vec[idx_not_nan])
    return mean_data
    
