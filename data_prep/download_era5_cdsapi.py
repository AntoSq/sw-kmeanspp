"""
Download ERA5 daily statistics from the Copernicus Climate Data Store
(dataset "ERA5 post-processed daily statistics on single levels from 1940 to
present", DOI 10.24381/cds.4991cf48 -- the manuscript's data source).

Requires a CDS API key in ~/.cdsapirc (never hardcoded or read from any other
location): https://cds.climate.copernicus.eu/how-to-api

Usage:
    python download_era5_cdsapi.py --variable tp --start 1991 --end 2020
    python download_era5_cdsapi.py --variable tp tmax sst mslp u10 v10 --start 1991 --end 2020
"""

import argparse
import os

import yaml

DATASET = "derived-era5-single-levels-daily-statistics"
HERE = os.path.dirname(os.path.abspath(__file__))


def download_year(client, variable_spec, year, target):
    """One request per year per variable: small, resumable, within CDS size limits.

    Deliberately requests the GLOBAL grid (no `area` restriction) so one
    download serves every domain in config/domains.yaml -- for the Global/MSLP
    domain this is required anyway; for the other 3 domains it trades some
    download size for not needing to re-fetch if a domain box ever changes.
    If a per-domain `area` request is ever added: CDS's `area` ordering is
    [North, West, South, East], the REVERSE of this repo's domains.yaml
    convention (lonW, lonE, latS, latN) -- an easy transcription bug.
    """
    client.retrieve(DATASET, {
        "product_type": "reanalysis",
        "variable": variable_spec['cds_variable'],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": variable_spec['daily_statistic'],
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
    }).download(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variable', nargs='+', required=True,
                         choices=['tp', 'tmax', 'sst', 'mslp', 'u10', 'v10'])
    parser.add_argument('--start', type=int, default=1991)
    parser.add_argument('--end', type=int, default=2020)
    parser.add_argument('--data-root', default=os.environ.get('SWKMEANSPP_DATA_ROOT', './data'))
    args = parser.parse_args()

    import cdsapi  # imported lazily so --help works without the package installed
    client = cdsapi.Client()

    with open(os.path.join(HERE, 'variables.yaml')) as f:
        variables = yaml.safe_load(f)

    for var in args.variable:
        var_dir = os.path.join(args.data_root, 'raw', var)
        os.makedirs(var_dir, exist_ok=True)
        for year in range(args.start, args.end + 1):
            target = os.path.join(var_dir, f'{var}_{year}.nc')
            if os.path.exists(target):
                print(f'{target} already exists, skipping')
                continue
            print(f'Downloading {var} {year} -> {target}')
            download_year(client, variables[var], year, target)


if __name__ == '__main__':
    main()
