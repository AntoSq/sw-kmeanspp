"""
ERA5 preprocessing: climatology, anomaly, domain box + land/sea mask, and the
gridpoint x time matrix clustering input -- pure xarray/numpy, no `cdo`
dependency required.
"""

import os
import pickle

import numpy as np
import xarray as xr
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swkmeanspp.masking import mask_for_clustering

HERE = os.path.dirname(os.path.abspath(__file__))


def load_raw(variable, data_root, start, end):
    files = [os.path.join(data_root, 'raw', variable, f'{variable}_{y}.nc') for y in range(start, end + 1)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing raw ERA5 files (run download_era5_cdsapi.py first): {missing}")
    return xr.open_mfdataset(files, combine='by_coords')


def compute_climatology(da, window=31):
    """Day-of-year running-mean climatology (pure xarray/numpy equivalent of `cdo -ydrunmean,31`),
    with circular wraparound across the year boundary."""
    doy = da['time'].dt.dayofyear
    clim_by_doy = da.groupby(doy).mean('time')
    n = clim_by_doy.sizes['dayofyear']
    arr = clim_by_doy.values
    half = window // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]], axis=0)
    smoothed = np.stack([padded[i:i + window].mean(axis=0) for i in range(n)], axis=0)
    return clim_by_doy.copy(data=smoothed)


def compute_anomaly(da, clim_by_doy):
    doy = da['time'].dt.dayofyear
    return da.groupby(doy) - clim_by_doy


def to_gridpoint_matrix(da):
    """(n_points, n_time) matrix + (n_points, 2) [lon, lat] pairs, dropping all-NaN gridpoints
    (masked-out land/sea or outside the domain box)."""
    brick = da.values  # (n_time, n_lat, n_lon)
    lat = da['lat'].values
    lon = da['lon'].values
    matrix = brick.reshape(brick.shape[0], -1).T
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lonlat_pairs = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    keep = ~np.isnan(matrix).all(axis=1)
    return matrix[keep], lonlat_pairs[keep]


def preprocess(variable, domain_name, period, data_root, lsm_path=None):
    """
    Full preprocessing pipeline for one variable/domain: load raw ->
    climatology -> anomaly -> domain mask -> gridpoint matrix -> pickle cache.

    Returns (data_cluster, lonlat_pairs, data_clim) and writes them under
    `{data_root}/processed/`.
    """
    with open(os.path.join(HERE, 'variables.yaml')) as f:
        var_specs = yaml.safe_load(f)
    with open(os.path.join(os.path.dirname(HERE), 'config', 'domains.yaml')) as f:
        domains = yaml.safe_load(f)

    var_spec = var_specs[variable]
    domain = domains[domain_name]

    ds = load_raw(variable, data_root, period['start'], period['end'])
    var_name = list(ds.data_vars)[0]
    da = ds[var_name] * var_spec['unit_conversion']

    clim = compute_climatology(da)
    anom = compute_anomaly(da, clim)

    lsm = xr.open_dataset(lsm_path or os.path.join(HERE, 'era5_lsm.nc')).round() if domain['land_or_sea'] != 'all' else None
    coord_extr = (domain['lonW'], domain['lonE'], domain['latS'], domain['latN'])

    anom_masked = mask_for_clustering(anom, coord_extr, lsm=lsm, land_or_sea=domain['land_or_sea'], exception=domain.get('exception'))
    clim_masked = mask_for_clustering(clim.rename({'dayofyear': 'time'}), coord_extr, lsm=lsm,
                                       land_or_sea=domain['land_or_sea'], exception=domain.get('exception'))

    data_cluster, lonlat_pairs = to_gridpoint_matrix(anom_masked)
    data_clim, _ = to_gridpoint_matrix(clim_masked)

    processed_dir = os.path.join(data_root, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    stem = f'{variable}_{domain_name}'
    with open(os.path.join(processed_dir, f'data_cluster_{stem}.pkl'), 'wb') as f:
        pickle.dump(data_cluster, f)
    with open(os.path.join(processed_dir, f'lonlatpairs_{stem}.pkl'), 'wb') as f:
        pickle.dump(lonlat_pairs, f)
    with open(os.path.join(processed_dir, f'data_clim_{stem}.pkl'), 'wb') as f:
        pickle.dump(data_clim, f)

    return data_cluster, lonlat_pairs, data_clim


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--start', type=int, default=1991)
    parser.add_argument('--end', type=int, default=2020)
    parser.add_argument('--data-root', default=os.environ.get('SWKMEANSPP_DATA_ROOT', './data'))
    args = parser.parse_args()
    preprocess(args.variable, args.domain, {'start': args.start, 'end': args.end}, args.data_root)
