"""
Generic experiment driver: one YAML config (domain + variable(s) + w_s/w_t +
K + seed + a `runs:` list of sweep/variant entries) in, cluster labels +
centroids + metrics + maps out -- see config/experiments/*.yaml for the
manuscript's 5 case studies.

Usage:
    python run_experiment.py --config ../config/experiments/tp_europe_sweep.yaml
    python run_experiment.py --config ../config/experiments/mslp_global.yaml --run no_background
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from swkmeanspp.config import load_experiment
from swkmeanspp.io import PathResolver
from swkmeanspp.pca import temporal_pca
from swkmeanspp.kmeans_raw import raw_kmeanspp
from swkmeanspp.init import kmeanspp_init
from swkmeanspp.swkmeans import run as run_swkmeans
from swkmeanspp.metrics import compute_silhouette_metrics, centroid_diagnostics
from swkmeanspp.plotting import map_clusters
from data_prep.preprocess_era5 import preprocess as preprocess_era5


def ensure_processed(variable, domain_name, period, paths):
    stem = f'{variable}_{domain_name}'
    cluster_path = paths.processed(f'data_cluster_{stem}.pkl')
    lonlat_path = paths.processed(f'lonlatpairs_{stem}.pkl')
    if os.path.exists(cluster_path) and os.path.exists(lonlat_path):
        with open(cluster_path, 'rb') as f:
            data_cluster = pickle.load(f)
        with open(lonlat_path, 'rb') as f:
            lonlat_pairs = pickle.load(f)
        return data_cluster, lonlat_pairs
    data_cluster, lonlat_pairs, _ = preprocess_era5(variable, domain_name, period, paths.data_root)
    return data_cluster, lonlat_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', default=None, help='Run only this run label (default: every run in the config).')
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    paths = PathResolver.from_yaml(
        os.path.join(REPO_ROOT, 'config', 'paths.yaml'),
        overrides={'data_root': args.data_root} if args.data_root else None)

    if cfg.get('enable_climatology_term'):
        print('!' * 70)
        print('WARNING: experimental.enable_climatology_term is set, but this driver')
        print('only wires the PUBLISHED (beta=0) distance formula end-to-end.')
        print('See swkmeanspp/climatology_experimental.py -- it is an unwired stub.')
        print('!' * 70)

    exp_name = cfg['name']
    variables = cfg['variables']
    k, seed = cfg['k'], cfg['seed']
    pca_cfg = cfg.get('pca', {})

    data_cluster, lonlat_pairs = {}, None
    for var in variables:
        dc, llp = ensure_processed(var, cfg['domain'], cfg['period'], paths)
        data_cluster[var] = dc
        lonlat_pairs = llp  # shared coords across variables (co-located grid)

    full_scores, n_retained = {}, {}
    for var in variables:
        fs, _, _, nr = temporal_pca(data_cluster[var], k, seed=seed,
                                     variance_threshold_pct=pca_cfg.get('variance_threshold_pct', 1.0))
        full_scores[var] = fs
        n_retained[var] = nr

    # Multi-variable alignment (manuscript Sect. 2.7): all variables share the same
    # PCA width, the max of each variable's own 1%-variance cutoff.
    width = max(n_retained.values())
    temporal_pca_scores = {var: full_scores[var][:, :min(width, full_scores[var].shape[1])] for var in variables}

    baselines = cfg.get('baselines', [])
    if 'kmeans_raw' in baselines:
        if len(variables) != 1:
            raise NotImplementedError('Raw k-means++ baseline is only wired for single-variable experiments.')
        var = variables[0]
        labels_raw, _ = raw_kmeanspp(data_cluster[var], k, seed=seed)
        cl_raw = pd.DataFrame({'cluster': labels_raw, 'nodes_lon': lonlat_pairs[:, 0], 'nodes_lat': lonlat_pairs[:, 1]})
        cl_raw.to_csv(paths.output(exp_name, f'{exp_name}_km_{k}.csv'))
        map_clusters(lonlat_pairs[:, 0], lonlat_pairs[:, 1], labels_raw,
                     f'{exp_name}: k-means++ (raw data)', paths.output(exp_name, f'map_{exp_name}_km.png'))

    centroids_pca, labels_pca = kmeanspp_init(temporal_pca_scores, k, seed=seed)
    if 'pca_kmeans' in baselines:
        cl_pca = pd.DataFrame({'cluster': labels_pca, 'nodes_lon': lonlat_pairs[:, 0], 'nodes_lat': lonlat_pairs[:, 1]})
        cl_pca.to_csv(paths.output(exp_name, f'{exp_name}_pcakm_{k}.csv'))
        map_clusters(lonlat_pairs[:, 0], lonlat_pairs[:, 1], labels_pca,
                     f'{exp_name}: pca-k-means++', paths.output(exp_name, f'map_{exp_name}_pcakm.png'))

    runs = cfg['runs']
    if args.run:
        runs = [r for r in runs if r['label'] == args.run]
        if not runs:
            raise ValueError(f'No run labeled {args.run!r} in {args.config}')

    for run_cfg in runs:
        label = run_cfg['label']
        max_iter = run_cfg.get('max_iter', 100)
        print(f'--- {exp_name} / {label} (w_s={run_cfg["w_s"]}, w_t={run_cfg["w_t"]}) ---')

        centroids_temporal, centroids_coords, labels, n_iter = run_swkmeans(
            temporal_pca_scores, lonlat_pairs, k,
            w_s=run_cfg['w_s'], w_t=run_cfg['w_t'], init_labels=labels_pca,
            max_iter=max_iter,
            allow_background_cluster=run_cfg.get('allow_background_cluster', True),
        )
        print(f'{label}: stopped at iteration {n_iter} (max_iter={max_iter})')

        cl_df = pd.DataFrame({'cluster': labels, 'nodes_lon': lonlat_pairs[:, 0], 'nodes_lat': lonlat_pairs[:, 1]})
        cl_df.to_csv(paths.output(exp_name, f'{exp_name}_swkm_{label}.csv'))
        pd.DataFrame(centroids_coords).to_csv(
            paths.output(exp_name, f'clcentroids_{exp_name}_swkm_{label}.csv'), header=None)

        avg_slh, geo_avg_slh, _ = compute_silhouette_metrics(cl_df, data_cluster[variables[0]])
        print(f'Temporal silhouette: {avg_slh:.3f}, spatial silhouette: {geo_avg_slh:.3f}')

        map_clusters(
            lonlat_pairs[:, 0], lonlat_pairs[:, 1], labels,
            f'{exp_name} / SW-k-means++ ({label})\ntemporal={avg_slh:.2f}, spatial={geo_avg_slh:.2f}',
            paths.output(exp_name, f'map_{exp_name}_swkm_{label}.png'), centroids=centroids_coords)

        for var in variables:
            diag = centroid_diagnostics([data_cluster[var][labels == c].mean(axis=0) for c in range(k)])
            diag.to_csv(paths.output(exp_name, f'centroid_diagnostics_{exp_name}_{label}_{var}.csv'))


if __name__ == '__main__':
    main()
