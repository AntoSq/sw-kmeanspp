"""
Standard k-means++ applied directly to the raw (non-PCA-reduced) per-gridpoint
series -- the manuscript's "k-means++ (raw data)" baseline, reported alongside
"pca-k-means++" and "SW-k-means++" (Sect. 3 preamble; Fig. 2a for TP Europe).
"""

import numpy as np
from sklearn.cluster import KMeans


def raw_kmeanspp(X, k, seed=0, n_init=1):
    """
    X: (n_points, n_time) raw per-gridpoint series for one variable.
    Returns (labels, centroids).
    """
    model = KMeans(n_clusters=k, random_state=seed, init='k-means++', n_init=n_init).fit(np.asarray(X))
    return model.labels_, model.cluster_centers_
