"""
SW-k-means++ core algorithm (manuscript Sects. 2.3-2.7), unified for any
number of variables: single-variable is just `len(temporal) == 1`, matching
how the manuscript itself generalizes the weighted-distance formula (Sect.
2.7 is literally Sect. 2.4's equation with a sum added over variables).

`init.kmeanspp_init` performs the preliminary standard k-means++ run the
manuscript requires (Sect. 2.5), so the first centroid snap below is
computed directly from its labels using the real `w_s`/`w_t`.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .background import detect_background_cluster, stabilize_spatial_distance
from .distances import normalize, weighted_distance_sq
from .geometry import haversine_vec, spatial_centroid


def _pairwise_temporal(X, centroids):
    return cdist(np.asarray(X), np.asarray(centroids), metric='euclidean')


def _pairwise_spatial(coords, centroid_coords):
    coords = np.asarray(coords)
    centroid_coords = np.asarray(centroid_coords)
    return haversine_vec(coords[:, 0:1], coords[:, 1:2], centroid_coords[None, :, 0], centroid_coords[None, :, 1])


def _assign(temporal, coords, centroids_temporal, centroids_coords, w_t, w_s, background_index):
    """One assignment step: returns (labels, combined_temporal_sq_by_nearest) for every gridpoint."""
    variables = list(temporal)
    k = len(centroids_coords)

    temporal_sq_norm = {}
    for var in variables:
        d = _pairwise_temporal(temporal[var], centroids_temporal[var])  # (n_points, k)
        temporal_sq_norm[var] = normalize(d) ** 2

    spat_d = _pairwise_spatial(coords, centroids_coords)  # (n_points, k)
    spat_sq_norm = normalize(spat_d) ** 2

    if background_index is not None:
        # Manuscript Sect. 2.6: neutralize the spatial term for gridpoints that would
        # otherwise join the background cluster (their temporal-only nearest cluster is it).
        temporal_only_sq = sum(w_t[var] * temporal_sq_norm[var] for var in variables)
        nearest_temporal = np.argmin(temporal_only_sq, axis=1)
        rows = np.where(nearest_temporal == background_index)[0]
        for i in rows:
            spat_sq_norm[i] = stabilize_spatial_distance(np.sqrt(spat_sq_norm[i]), background_index) ** 2

    combined_sq = weighted_distance_sq(temporal_sq_norm, spat_sq_norm, w_t, w_s)
    labels = np.argmin(combined_sq, axis=1)
    return labels


def _update_centroids(temporal, coords, labels, k, w_t, w_s):
    """Centroid update (manuscript Sect. 2.5): per-variable temporal mean, then the
    two-step spatial centroid (provisional median -> snap to nearest gridpoint by
    weighted distance)."""
    variables = list(temporal)
    n_components = {var: np.asarray(temporal[var]).shape[1] for var in variables}

    new_centroids_temporal = {var: np.zeros((k, n_components[var])) for var in variables}
    new_centroids_coords = np.zeros((k, 2))

    for j in range(k):
        members = np.where(labels == j)[0]
        if len(members) == 0:
            continue

        temporal_centroid_j = {}
        temporal_sq_norm = {}
        for var in variables:
            X = np.asarray(temporal[var])
            centroid = X[members].mean(axis=0)
            temporal_centroid_j[var] = centroid
            d = np.linalg.norm(X[members] - centroid, axis=1)
            temporal_sq_norm[var] = normalize(d) ** 2

        provisional_lon, provisional_lat = spatial_centroid(np.asarray(coords)[members])
        spat_d = haversine_vec(np.asarray(coords)[members, 0], np.asarray(coords)[members, 1],
                                provisional_lon, provisional_lat)
        spat_sq_norm = normalize(spat_d) ** 2

        combined_sq = weighted_distance_sq(temporal_sq_norm, spat_sq_norm, w_t, w_s)
        snap_index = members[np.argmin(combined_sq)]

        for var in variables:
            new_centroids_temporal[var][j] = temporal_centroid_j[var]
        new_centroids_coords[j] = np.asarray(coords)[snap_index]

    return new_centroids_temporal, new_centroids_coords


def run(temporal, coords, k, w_s, w_t, init_labels, max_iter=100,
        allow_background_cluster=True, background_threshold=0.5, verbose=False):
    """
    SW-k-means++ (manuscript Sects. 2.3-2.7).

    temporal: {variable: (n_points, n_components) PCA-reduced temporal data}.
    coords: (n_points, 2) array of [lon, lat], shared across variables.
    k: number of clusters.
    w_s: spatial weight. w_t: {variable: weight}. sum(w_t.values()) + w_s == 1.
    init_labels: labels from a preliminary standard k-means++ run
      (swkmeanspp.init.kmeanspp_init), required.
    allow_background_cluster: whether to apply the >50%-cluster stabilization
      (manuscript Sect. 2.6).

    Returns (centroids_temporal, centroids_coords, labels, n_iterations).
    """
    variables = list(temporal)
    labels = np.asarray(init_labels)
    centroids_temporal, centroids_coords = _update_centroids(temporal, coords, labels, k, w_t, w_s)

    prev_labels = None
    background_index = None
    iteration = 0
    for iteration in range(max_iter):
        if verbose:
            print(f'\nIteration {iteration}')

        labels = _assign(temporal, coords, centroids_temporal, centroids_coords, w_t, w_s, background_index)

        # Convergence: stop when cluster assignments are unchanged from the previous iteration
        # (manuscript Sect. 2.3: "terminated when no changes in the cluster assignments ...").
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            if verbose:
                print(f'Finished at iteration {iteration}')
            break
        prev_labels = labels

        background_index = detect_background_cluster(labels, k, background_threshold) if allow_background_cluster else None
        if verbose and background_index is not None:
            print(f'Background cluster: {background_index}')

        centroids_temporal, centroids_coords = _update_centroids(temporal, coords, labels, k, w_t, w_s)

    return centroids_temporal, centroids_coords, labels, iteration
