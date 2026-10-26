"""
Evaluation metrics (manuscript Sect. 2 evaluation / Appendix): temporal and
spatial silhouette (always computed on the original, non-PCA-reduced
per-gridpoint series), RMS, lag-1 autocorrelation, 3-band power-spectrum
fractions, kurtosis of day-to-day increments, correlation maps, medoid
(best-corr-gp), and the area-weighted "0.1corr share".
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import welch
from sklearn.metrics import silhouette_score, silhouette_samples


def corr(a, b):
    """Normalized dot-product correlation between two equal-length series."""
    a = a - a.mean()
    b = b - b.mean()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def silhouette_from_df(df, X, cluster_col='cluster', sample_size=None, random_state=42, return_samples=False):
    """Silhouette score for clustering results stored in a DataFrame (rows of X must match df's rows)."""
    labels = df[cluster_col].values
    silhouette_avg = silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
    if return_samples:
        return silhouette_avg, silhouette_samples(X, labels)
    return silhouette_avg


def geographic_silhouette_score(df, X, cluster_col='cluster', sample_size=None, random_state=42, return_samples=False):
    """Silhouette score for geographic coordinates (X = [lat, lon] in degrees) using haversine distance."""
    labels = df[cluster_col].values
    X_rad = np.radians(X)
    silhouette_avg = silhouette_score(X_rad, labels, metric='haversine', sample_size=sample_size, random_state=random_state)
    if return_samples:
        return silhouette_avg, silhouette_samples(X_rad, labels, metric='haversine')
    return silhouette_avg


def compute_silhouette_metrics(clusters_df, data_cluster, sample_size=10000, random_state=42):
    """
    Temporal and spatial silhouette -- always computed on the original
    (non-PCA-reduced) per-gridpoint series `data_cluster`, regardless of
    what space the upstream clustering algorithm operated in.

    Returns: avg_slh (temporal), geo_avg_slh (spatial), and a per-cluster
    average (temporal + spatial silhouette, cluster fraction).
    """
    avg_slh = silhouette_from_df(clusters_df, data_cluster, sample_size=sample_size, random_state=random_state)
    geo_avg_slh = geographic_silhouette_score(
        clusters_df, clusters_df[['nodes_lat', 'nodes_lon']].to_numpy(), sample_size=sample_size, random_state=random_state)

    sub_clusters_df = clusters_df.sample(n=min(sample_size, len(clusters_df)), random_state=random_state)
    sub_indices = [clusters_df.index.get_loc(idx) for idx in sub_clusters_df.index]
    sub_data_cluster = np.asarray(data_cluster)[sub_indices]

    _, slh_values = silhouette_from_df(sub_clusters_df, sub_data_cluster, sample_size=sample_size, return_samples=True, random_state=random_state)
    _, geo_slh_values = geographic_silhouette_score(
        sub_clusters_df, sub_clusters_df[['nodes_lat', 'nodes_lon']].to_numpy(),
        sample_size=sample_size, return_samples=True, random_state=random_state)

    sub_clusters_df = sub_clusters_df.copy()
    sub_clusters_df['silh_temp'] = slh_values
    sub_clusters_df['silh_spat'] = geo_slh_values
    sub_cl_silh_avg = sub_clusters_df.groupby('cluster')[['silh_temp', 'silh_spat']].mean()
    sub_cl_silh_avg['cluster_frac'] = sub_clusters_df['cluster'].value_counts(normalize=True).round(2)
    return avg_slh, geo_avg_slh, sub_cl_silh_avg


def centroid_diagnostics(centroids, fs=1.0):
    """
    Per-centroid diagnostics: RMS, intermittency, lag-1 autocorrelation,
    3-band spectral power fractions (low <0.1, synoptic 0.1-0.3, high >=0.3
    day^-1), and kurtosis of first differences.
    """
    records = []
    for k, c in enumerate(centroids):
        c = np.asarray(c).astype(float)
        c = c - np.mean(c)

        rms = np.sqrt(np.mean(c ** 2))
        denom = np.mean(np.abs(c)) ** 2
        intermittency = np.mean(c ** 2) / denom if denom != 0 else np.nan
        acf1 = np.corrcoef(c[:-1], c[1:])[0, 1] if len(c) > 1 else np.nan

        try:
            f, P = welch(c, fs=fs, nperseg=min(256, len(c)))
            total_power = P.sum()
            low_freq = P[f < 0.1].sum() / total_power
            synoptic = P[(f >= 0.1) & (f < 0.3)].sum() / total_power
            high_freq = P[f >= 0.3].sum() / total_power
        except Exception:
            low_freq = synoptic = high_freq = np.nan

        kurt = kurtosis(np.diff(c), fisher=True, bias=False) if len(c) > 2 else np.nan

        records.append({
            "cluster": k, "RMS": rms, "Intermittency": intermittency, "ACF1": acf1,
            "LowFreq": low_freq, "Synoptic": synoptic, "HighFreq": high_freq, "Kurtosis_dC": kurt,
        })
    return pd.DataFrame(records).set_index("cluster")


def _compute_corr_map_chunked(X, ref, chunk_size=10000):
    """Correlation between each row of X (n_points, n_time) and ref (n_time,), chunked to bound memory."""
    ref = ref.astype(np.float32, copy=False)
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        ref_norm = 1.0
    ref = ref / ref_norm

    n_points = X.shape[0]
    out = np.empty(n_points, dtype=np.float32)
    for i in range(0, n_points, chunk_size):
        chunk = X[i:i + chunk_size].astype(np.float32, copy=False).copy()
        chunk -= chunk.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk /= norms
        out[i:i + chunk_size] = np.dot(chunk, ref)
    return out


def corr01_share(centroid_corr, medoid_corr, lat, in_cluster=None, threshold=0.1):
    """
    Area-weighted share (%) of gridpoints -- restricted to `in_cluster` if
    given, else all gridpoints in the arrays -- whose correlation to both
    the centroid and the medoid (best-corr-gp) exceeds `threshold`.
    Weighted by cos(latitude) so high-latitude gridpoints (denser on a
    regular lon/lat grid) don't get over-counted.
    """
    if in_cluster is None:
        in_cluster = np.ones_like(centroid_corr, dtype=bool)
    area_weight = np.cos(np.deg2rad(lat))
    both_above = (centroid_corr > threshold) & (medoid_corr > threshold) & in_cluster
    weight_in_cluster = area_weight[in_cluster].sum()
    if weight_in_cluster <= 0:
        return np.nan
    return 100.0 * area_weight[both_above].sum() / weight_in_cluster


def build_cluster_correlation_df(X, labels, lonlat_pairs):
    """
    For every gridpoint: correlation to each cluster centroid
    ('cluster_{cl}'), correlation to each cluster's medoid/best-corr-gp
    ('medoid_corr_{cl}', the in-cluster gridpoint most correlated to the
    centroid), plus the cluster label ('cluster'). Always computed on the
    original (non-PCA-reduced) per-gridpoint series X.

    X: (n_points, n_time) array for one variable.
    labels: cluster label per gridpoint.
    lonlat_pairs: array/DataFrame with columns [lon, lat].
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    lonlat_pairs = np.asarray(lonlat_pairs)
    corr_df = pd.DataFrame({
        'nodes_lon': lonlat_pairs[:, 0],
        'nodes_lat': lonlat_pairs[:, 1],
        'cluster': labels,
    })
    centroids, medoids = [], []
    for cl in np.unique(labels):
        in_cluster = (labels == cl)
        centroid = X[in_cluster].mean(axis=0)
        centroids.append(centroid)
        corr_df[f'cluster_{cl}'] = _compute_corr_map_chunked(X, centroid)

        corr_to_centroid = corr_df.loc[in_cluster, f'cluster_{cl}'].values
        medoid = X[in_cluster][np.argmax(corr_to_centroid)]
        medoids.append(medoid)
        corr_df[f'medoid_corr_{cl}'] = _compute_corr_map_chunked(X, medoid)

    return corr_df, centroids, medoids
