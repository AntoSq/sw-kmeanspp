# Method notes

Implements Squintu et al. (submitted to GMD), Sects. 2.1–2.7. This document
collects formulas and reproducibility notes not already covered in the
module docstrings (`swkmeanspp/distances.py`, `swkmeanspp/swkmeans.py`,
`swkmeanspp/background.py`, `swkmeanspp/init.py`, `swkmeanspp/pca.py`).

## Formulas

- Temporal PCA (Sect. 2.2): retain every EOF individually explaining >= 1%
  of variance. `swkmeanspp/pca.py::temporal_pca`.
- Haversine spatial distance (Sect. 2.4, cited not derived in the
  manuscript): `swkmeanspp/geometry.py::haversine` / `haversine_vec`, Earth
  radius 6371 km.
- Per-gridpoint normalization, same form for spatial and every temporal
  term: `d_tilde = (d - d.min()) / d.mean()`. `swkmeanspp/distances.py::normalize`.
- Weighted distance: `wd^2 = sum_v(w_t,v * d_t,v_tilde^2) + w_s * d_s_tilde^2`,
  `sum(w_t,v) + w_s = 1`. Single-variable is just one term in the sum.
  `swkmeanspp/distances.py::weighted_distance_sq`.
- Spatial centroid (Sect. 2.5): provisional = (median latitude, median
  longitude within the dominant 15-degree band, circular wraparound at
  ±180°); snapped to the cluster gridpoint minimizing the weighted distance
  to (temporal centroid, provisional spatial point).
  `swkmeanspp/geometry.py::spatial_centroid`, `swkmeanspp/swkmeans.py::_update_centroids`.
- Background-cluster stabilization (Sect. 2.6): triggers when a cluster
  holds > 50% of gridpoints; that cluster's spatial distance is replaced by
  the minimum distance among the other clusters' centroids, for gridpoints
  whose temporal-only nearest cluster is the background one.
  `swkmeanspp/background.py`.
- Convergence (Sect. 2.3): stop when cluster assignments are unchanged
  between two successive iterations.

## Reproducibility

- `init.kmeanspp_init` (sklearn `KMeans(init='k-means++')` on PCA-reduced
  data, fixed `random_state`) is the initializer. Given the same library
  versions, the same experiment config reproduces the same labels and
  centroids deterministically.
- sklearn's `KMeans` is not guaranteed bit-reproducible across different
  scikit-learn/BLAS versions even with a fixed seed — the archived output
  CSVs/PNGs (not a from-scratch re-run on a different machine) are the
  ground truth for reproducing a specific published figure exactly.
