"""
Background-cluster stabilization (manuscript Sect. 2.6): when a cluster
holds more than half of the domain's gridpoints, its spatial-distance
penalty is neutralized for gridpoints that would otherwise be assigned to
it, so a single very large, spatially-spread-out cluster isn't artificially
split just because of its size.
"""

import numpy as np


def detect_background_cluster(labels, k, threshold=0.5):
    """
    Return the label of the background cluster (a cluster covering strictly
    more than `threshold` of all gridpoints), or None if there isn't one.
    Per manuscript Sect. 2.6, the stabilization "is no longer valid if ... the
    background cluster size falls below the threshold" -- i.e. the trigger
    is strict `>` and the de-trigger is `<=`, both handled by this one check
    being re-evaluated every iteration from the current labels.
    """
    unique_values, counts = np.unique(labels, return_counts=True)
    is_background = counts > threshold * len(labels)
    if np.any(is_background):
        return int(unique_values[is_background][0])
    return None


def stabilize_spatial_distance(spat_distances, background_index):
    """
    Replace the background cluster's spatial distance (for one gridpoint)
    with the minimum distance among the OTHER clusters' centroids, per
    manuscript Sect. 2.6: "replacing this value with the minimum distance
    among all the geographical centroids, thereby neutralizing the spatial
    contribution for the background cluster gridpoints" -- so the gridpoint
    is only kept out of the background cluster if a competing cluster is
    closer in temporal distance (left unchanged by this function).

    spat_distances: array of length K (one per cluster) for one gridpoint.
    background_index: index of the background cluster in that array.
    Returns a new array (input is not mutated).
    """
    spat_distances = np.asarray(spat_distances, dtype=float).copy()
    other = np.delete(spat_distances, background_index)
    spat_distances[background_index] = min(spat_distances[background_index], other.min())
    return spat_distances
