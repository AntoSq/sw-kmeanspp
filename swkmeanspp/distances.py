"""
Distance normalization and the weighted-distance combination (manuscript
Sects. 2.4 and 2.7).

A single `normalize` function is used for every distance term (spatial and
every variable's temporal term), in both the assignment step and the
centroid-update ("snap to nearest gridpoint") step.
"""

import numpy as np


def normalize(d):
    """
    Normalization of a distance vector (manuscript Eq. in Sect. 2.4):
    d_tilde = (d - d.min()) / d.mean(), the SAME form for both the spatial
    and every temporal distance term ("a similar procedure is applied to the
    temporal distances").

    Used in two contexts with the same formula: (1) the assignment step,
    where `d` is one gridpoint's distances to each of the K centroids
    (normalized across centroids); (2) the centroid-update "snap to nearest
    gridpoint" step, where `d` is one cluster's members' distances to a
    single candidate center (normalized across cluster members). `d` may be
    1D (a single such vector) or 2D with each row normalized independently
    (vectorized assignment step, shape (n_points, k)).
    """
    d = np.asarray(d, dtype=float)
    return (d - d.min(axis=-1, keepdims=True)) / d.mean(axis=-1, keepdims=True)


def weighted_distance_sq(temporal_sq_norm, spatial_sq_norm, w_t, w_s):
    """
    Squared weighted distance (manuscript Eq. in Sect. 2.4/2.7), omitting the
    final sqrt since only relative order matters for argmin-based assignment
    and centroid snapping:

        wd^2 = sum_v( w_t[v] * temporal_sq_norm[v] ) + w_s * spatial_sq_norm

    temporal_sq_norm: {variable: array_of_K} of normalize(d_t)**2 per variable.
    spatial_sq_norm: array_of_K of normalize(d_s)**2.
    w_t: {variable: weight}, w_s: weight. sum(w_t.values()) + w_s == 1.
    """
    total = w_s * spatial_sq_norm
    for var, d_sq in temporal_sq_norm.items():
        total = total + w_t[var] * d_sq
    return total


def weighted_distance(temporal_sq_norm, spatial_sq_norm, w_t, w_s):
    """sqrt of weighted_distance_sq -- the manuscript's wd as literally written."""
    return np.sqrt(weighted_distance_sq(temporal_sq_norm, spatial_sq_norm, w_t, w_s))
