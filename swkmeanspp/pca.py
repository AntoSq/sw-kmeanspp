"""Temporal PCA (manuscript Sect. 2.2): retain every EOF individually explaining
>= 1% of the total variance."""

import numpy as np
from sklearn.decomposition import PCA


def temporal_pca(X, k, variance_threshold_pct=1.0, n_components_factor=5, seed=0):
    """
    X: (n_points, n_time) raw per-gridpoint series for one variable.
    k: number of clusters (used only to size the initial, generous PCA fit).

    Returns (full_scores, pca, explained_variance_pct, n_retained), where
    `full_scores` has `k * n_components_factor` columns (or fewer if the
    data doesn't support that many), `n_retained` is how many of those
    columns individually explain >= variance_threshold_pct of the variance
    (manuscript Sect. 2.2) -- callers typically use `full_scores[:, :n_retained]`,
    except a multi-variable joint run, which aligns all variables to the
    SAME width (max of each variable's own `n_retained`) before clustering,
    per manuscript Sect. 2.7; `full_scores` makes that re-slicing possible
    without refitting PCA.
    """
    n_components = min(k * n_components_factor, min(np.asarray(X).shape))
    pca = PCA(n_components=n_components, random_state=seed)
    full_scores = pca.fit_transform(np.asarray(X))

    explained = pca.explained_variance_ratio_ * 100
    idx = np.where(explained >= variance_threshold_pct)[0]
    last_idx = idx[-1] if len(idx) > 0 else len(explained) - 1
    n_retained = last_idx + 1

    return full_scores, pca, explained, n_retained
