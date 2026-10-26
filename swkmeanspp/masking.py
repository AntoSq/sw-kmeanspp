"""
Domain box, land/sea mask, and named exclusion regions (manuscript Table 1):
Europe, North Atlantic (minus Mediterranean), Mediterranean (minus Bay of
Biscay and the Black Sea), Global.
"""

import xarray as xr


def apply_land_sea_mask(targetxr, lsm, kind='land'):
    """Mask out land or sea gridpoints of `targetxr` using `lsm` (era5_lsm.nc)."""
    lsm_mask = lsm.interp(longitude=targetxr['lon'], latitude=targetxr['lat'], method='nearest')
    lsm_mask = lsm_mask.fillna(0).squeeze('time').astype(bool)
    if kind == 'land':
        return targetxr.where(lsm_mask['lsm'])
    if kind == 'sea':
        return targetxr.where(~lsm_mask['lsm'])
    return targetxr


def _apply_exception(obs_masked, exception):
    if exception == 'NoBizcayaNoBlackSea':
        # Table 1's Mediterranean domain, excluding the Bay of Biscay and the Black Sea.
        return obs_masked.where(
            (((obs_masked.lat < 43) | (obs_masked.lon > 0))) &
            (((obs_masked.lat <= 40) | (obs_masked.lon <= 27))), drop=True)
    if exception == 'NoMediterranean':
        # Table 1's North Atlantic domain: "Removal of Mediterranean".
        return obs_masked.where(
            (((obs_masked.lat < 30) | (obs_masked.lat > 47)) |
             ((obs_masked.lon < -5) | (obs_masked.lon > 42))) |
            ((obs_masked.lat > 43) & (obs_masked.lon < 0)),  # keep the Bay of Biscay itself
            drop=True)
    raise ValueError(f"Unknown domain exception: {exception!r}")


def mask_for_clustering(obs, coord_extr, lsm=None, land_or_sea='all', exception=None):
    """
    Apply a domain box, an optional named exception, and an optional
    land/sea mask to an xarray object.

    obs: xarray.Dataset/DataArray to mask.
    coord_extr: (lonW, lonE, latS, latN). Use the full globe
      (-180, 180, -90, 90) for the Global domain (no box cut needed).
    lsm: xarray.Dataset land-sea mask (required unless land_or_sea == 'all').
    land_or_sea: 'land' | 'sea' | 'all'.
    exception: None | 'NoMediterranean' | 'NoBizcayaNoBlackSea'.
    """
    lonW, lonE, latS, latN = coord_extr
    obs_masked = obs.sel(lat=slice(latS, latN), lon=slice(lonW, lonE))

    if exception:
        obs_masked = _apply_exception(obs_masked, exception)

    if land_or_sea != 'all':
        obs_masked = apply_land_sea_mask(obs_masked, lsm, kind=land_or_sea)

    return obs_masked
