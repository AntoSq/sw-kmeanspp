"""Tests for the manuscript's distance-normalization formula (Sect. 2.4):
the SAME (d-d_min)/d_mean form is used for the spatial term and every
temporal term, in both the assignment step and the centroid-update step."""

import numpy as np

from swkmeanspp.distances import normalize, weighted_distance_sq


def test_normalize_matches_manuscript_formula_1d():
    d = np.array([1.0, 2.0, 4.0, 8.0])
    expected = (d - d.min()) / d.mean()
    np.testing.assert_allclose(normalize(d), expected)


def test_normalize_matches_manuscript_formula_2d_rows():
    d = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 5.0]])
    expected = np.array([(row - row.min()) / row.mean() for row in d])
    np.testing.assert_allclose(normalize(d), expected)


def test_normalize_gives_zero_to_the_closest_centroid():
    # Manuscript Sect. 2.4: the scaling "gives null distance to the closest cluster".
    d = np.array([3.0, 7.0, 1.0, 9.0])
    normalized = normalize(d)
    assert normalized[np.argmin(d)] == 0.0


def test_weighted_distance_sq_single_variable():
    temporal_sq = {'tp': np.array([0.0, 1.0, 4.0])}
    spatial_sq = np.array([1.0, 0.0, 1.0])
    w_t, w_s = {'tp': 0.95}, 0.05
    expected = w_t['tp'] * temporal_sq['tp'] + w_s * spatial_sq
    np.testing.assert_allclose(weighted_distance_sq(temporal_sq, spatial_sq, w_t, w_s), expected)


def test_weighted_distance_sq_multi_variable_sums_over_variables():
    temporal_sq = {'u10': np.array([1.0, 0.0]), 'v10': np.array([0.0, 1.0])}
    spatial_sq = np.array([0.5, 0.5])
    w_t, w_s = {'u10': 0.475, 'v10': 0.475}, 0.05
    expected = 0.475 * temporal_sq['u10'] + 0.475 * temporal_sq['v10'] + 0.05 * spatial_sq
    np.testing.assert_allclose(weighted_distance_sq(temporal_sq, spatial_sq, w_t, w_s), expected)
