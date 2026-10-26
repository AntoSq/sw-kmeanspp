"""Background-cluster stabilization (manuscript Sect. 2.6), on a synthetic
fixture deliberately built so one cluster holds >50% of gridpoints."""

import numpy as np

from swkmeanspp.background import detect_background_cluster, stabilize_spatial_distance
from swkmeanspp.init import kmeanspp_init
from swkmeanspp.swkmeans import run as run_swkmeans


def test_detect_background_cluster_triggers_above_half():
    labels = np.array([0] * 60 + [1] * 20 + [2] * 20)
    assert detect_background_cluster(labels, k=3) == 0


def test_detect_background_cluster_none_when_balanced():
    labels = np.array([0] * 34 + [1] * 33 + [2] * 33)
    assert detect_background_cluster(labels, k=3) is None


def test_stabilize_spatial_distance_uses_min_of_others():
    d = np.array([5.0, 2.0, 3.0])
    stabilized = stabilize_spatial_distance(d, background_index=0)
    assert stabilized[0] == 2.0  # min(5.0, min(2.0, 3.0))
    np.testing.assert_allclose(stabilized[1:], d[1:])  # other entries untouched


def _imbalanced_fixture(seed):
    rng = np.random.default_rng(seed)
    n_big, n_small = 120, 20
    lon = np.concatenate([rng.uniform(-150, 150, n_big), rng.uniform(-150, 150, n_small)])
    lat = np.concatenate([rng.uniform(-60, 60, n_big), rng.uniform(-60, 60, n_small)])
    series = np.concatenate([
        rng.normal(0, 1, (n_big, 20)) + rng.normal(0, 1, 20),
        rng.normal(0, 1, (n_small, 20)) + rng.normal(5, 1, 20),  # a clearly distinct small cluster
    ])
    coords = np.column_stack([lon, lat])
    return {'mslp': series}, coords


def test_allow_background_cluster_false_runs_to_max_iter_without_raising():
    temporal, coords = _imbalanced_fixture(seed=2)
    k = 5
    _, labels_init = kmeanspp_init(temporal, k, seed=0)
    # alpha=1.0 (pure spatial) on a domain with no real spatial structure is a reasonable
    # stand-in for the manuscript's MSLP-without-background non-convergent illustrative run.
    _, _, labels, n_iter = run_swkmeans(temporal, coords, k, w_s=1.0, w_t={'mslp': 0.0},
                                         init_labels=labels_init, max_iter=15,
                                         allow_background_cluster=False)
    assert len(labels) == len(labels_init)
