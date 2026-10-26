"""Geometry helpers: haversine distance and the circular-aware longitude median."""

from math import radians, sin, cos, sqrt, atan2

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between two (lon, lat) points in degrees."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))  # clamp against math-domain errors from float rounding
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance (km); any broadcastable combination of scalars/arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def find_first_true_after_false(arr):
    """Index of the first True that follows a False in a boolean array (or None)."""
    if arr[0] and not arr[-1]:
        for i in range(len(arr) - 1, -1, -1):
            if not arr[i]:
                return np.argmax(arr[i:])
    for i in range(1, len(arr)):
        if arr[i] and not arr[i - 1]:
            return i
    return None


def median_longitudes(lon_array):
    """
    Median longitude of a set of points, robust to antimeridian wraparound
    (manuscript Sect. 2.5): histograms into 15-degree bins, keeps the
    dominant contiguous band of bins, and takes the median within that band
    (shifted to avoid wrapping).
    """
    bins = np.arange(-180, 180 + 15, 15)
    counts, bin_edges = np.histogram(lon_array, bins=bins)

    large_counts_bands = [counts >= np.max(counts) * 0.66][0]
    edgesW = bins[0:len(bins) - 1][large_counts_bands]
    edgesE = bins[1:len(bins)][large_counts_bands]
    if len(edgesE) == len(bins) - 1:
        return 0

    minbin = bins[0:len(bins) - 1][find_first_true_after_false(large_counts_bands)]

    large_counts_lons = []
    for e in range(len(edgesW)):
        lons_to_append = lon_array[(lon_array >= edgesW[e]) & (lon_array <= edgesE[e])]
        large_counts_lons.append(lons_to_append)
    large_counts_lons = np.concatenate(large_counts_lons)

    large_counts_lons_shifted = large_counts_lons - minbin
    large_counts_lons_shifted[large_counts_lons_shifted < 0] += 360
    medlon = np.median(large_counts_lons_shifted) + minbin
    if medlon <= -180:
        medlon += 360
    if medlon >= 180:
        medlon -= 360
    return medlon


def spatial_centroid(coords):
    """
    Provisional spatial centroid of a cluster's gridpoints (manuscript Sect. 2.5,
    step 1): latitude = plain median, longitude = median within the dominant
    15-degree band (median_longitudes, handles antimeridian wraparound).

    coords: (n_points, 2) array of [lon, lat].
    Returns (lon, lat).
    """
    coords = np.asarray(coords)
    lat = np.median(coords[:, 1])
    lon = median_longitudes(coords[:, 0])
    return lon, lat
