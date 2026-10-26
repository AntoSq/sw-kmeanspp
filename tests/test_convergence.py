"""SW-k-means++ converges (assignment-based criterion, manuscript Sect. 2.3)
well before max_iter on a well-separated synthetic problem, and running it
twice on identical inputs gives identical results (sklearn k-means++ with a
fixed seed is deterministic)."""

import numpy as np

from swkmeanspp.init import kmeanspp_init
from swkmeanspp.swkmeans import run as run_swkmeans


def make_blobs(seed, n_per_cluster=60, k=4, n_time=30):
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
    return {'tp': np.array(series)}, coords


def test_converges_before_max_iter():
    temporal, coords = make_blobs(seed=0)
    k = 4
    centroids_pca, labels_init = kmeanspp_init(temporal, k, seed=0)
    _, _, labels, n_iter = run_swkmeans(temporal, coords, k, w_s=0.05, w_t={'tp': 0.95},
                                        init_labels=labels_init, max_iter=60)
    assert n_iter < 59
    assert len(np.unique(labels)) == k


def test_deterministic_with_fixed_seed():
    temporal, coords = make_blobs(seed=1)
    k = 4
    _, labels_init = kmeanspp_init(temporal, k, seed=0)
    _, coords_a, labels_a, _ = run_swkmeans(temporal, coords, k, w_s=0.05, w_t={'tp': 0.95},
                                             init_labels=labels_init, max_iter=60)
    _, coords_b, labels_b, _ = run_swkmeans(temporal, coords, k, w_s=0.05, w_t={'tp': 0.95},
                                             init_labels=labels_init, max_iter=60)
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_allclose(coords_a, coords_b)
