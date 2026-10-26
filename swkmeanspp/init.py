"""
Initialization of SW-k-means++ (manuscript Sect. 2.5): "the spatially-weighted
clustering is initialized with the centroids obtained with a preliminary run
of standard k-means++ (i.e. without weighted spatial distances)."

sklearn's `KMeans(init='k-means++')` on the PCA-reduced temporal data is the
initializer used here.
"""

import numpy as np
from sklearn.cluster import KMeans


def kmeanspp_init(temporal_pca, k, seed=0, n_init=1):
    """
    Preliminary standard k-means++ run on the PCA-reduced temporal data.

    temporal_pca: {variable: (n_points, n_components) array} -- single or
    multi-variable; for the multi-variable case, the preliminary run
    concatenates all variables' PCA components so the initial partition
    already reflects every variable jointly.

    Returns {variable: (k, n_components) centroid array}, labels (shared
    across variables).
    """
    variables = list(temporal_pca)
    combined = np.concatenate([np.asarray(temporal_pca[v]) for v in variables], axis=1)
    model = KMeans(n_clusters=k, random_state=seed, init='k-means++', n_init=n_init).fit(combined)
    labels = model.labels_

    centroids = {}
    for var in variables:
        X = np.asarray(temporal_pca[var])
        centroids[var] = np.array([X[labels == c].mean(axis=0) for c in range(k)])
    return centroids, labels
