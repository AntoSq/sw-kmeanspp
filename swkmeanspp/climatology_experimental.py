"""
Experimental climatology-weighting term. The manuscript's weighted-distance
formula (Sects. 2.4, 2.7) is a 2-term `w_t`/`w_s` combination only; this
module provides an additional, optional climatology term for extending the
method, not used in any of the published results. Setting `beta=0` (the
default) reproduces the published method exactly.
"""

from sklearn.decomposition import PCA


def climatology_pca(data_clim, k, seed=0):
    """PCA-reduce the climatology array (k components)."""
    pca = PCA(n_components=k, random_state=seed)
    return pca.fit_transform(data_clim), pca
