"""The unified swkmeans.run generalizes single- and multi-variable clustering
into one implementation (manuscript Sect. 2.7 is Sect. 2.4's equation with a
sum over variables) -- this checks that generalization is consistent: running
with one variable duplicated as two identical variables (equal weights
summing to the single-variable weight) reproduces the single-variable
result."""

import numpy as np

from swkmeanspp.init import kmeanspp_init
from swkmeanspp.swkmeans import run as run_swkmeans


def make_blobs(seed, n_per_cluster=50, k=4, n_time=25):
    rng = np.random.default_rng(seed)
    lon, lat, series = [], [], []
    for c in range(k):
        lon0, lat0 = rng.uniform(-150, 150), rng.uniform(-50, 50)
        base = rng.normal(0, 1, n_time)
        for _ in range(n_per_cluster):
            lon.append(lon0 + rng.normal(0, 1))
            lat.append(np.clip(lat0 + rng.normal(0, 1), -89, 89))
            series.append(base + rng.normal(0, 0.1, n_time))
    coords = np.column_stack([lon, lat])
    return np.array(series), coords


def test_single_variable_matches_duplicated_two_variable():
    X, coords = make_blobs(seed=3)
    k = 4

    temporal_1v = {'tp': X}
    _, labels_init_1v = kmeanspp_init(temporal_1v, k, seed=0)
    _, coords_1v, labels_1v, _ = run_swkmeans(
        temporal_1v, coords, k, w_s=0.05, w_t={'tp': 0.95}, init_labels=labels_init_1v, max_iter=60)

    temporal_2v = {'tp_a': X, 'tp_b': X}
    _, labels_init_2v = kmeanspp_init(temporal_2v, k, seed=0)
    _, coords_2v, labels_2v, _ = run_swkmeans(
        temporal_2v, coords, k, w_s=0.05, w_t={'tp_a': 0.475, 'tp_b': 0.475},
        init_labels=labels_init_2v, max_iter=60)

    np.testing.assert_array_equal(labels_1v, labels_2v)
    np.testing.assert_allclose(coords_1v, coords_2v)
