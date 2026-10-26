# sw-kmeanspp

Spatially weighted k-means++ (SW-k-means++) for the regionalization of
climate data — code accompanying:

> Squintu, A., Pérez-Aracil, J., McAdam, R., Fella, E., Iakovidis, I., and
> Scoccimarro, E.: Spatially Weighted k-means++: an updated regionalization
> approach for climate data, *Geoscientific Model Development* (submitted),
> 2026. Paper DOI: `<PAPER_DOI>` (to be filled in once published).

## Citation

If you use this code, please cite it as:

```bibtex
@misc{squintu2026swkmeanscode,
  author={Squintu, Antonello and P{\'e}rez-Aracil, Jorge and McAdam, Ronan and Fella, Eugenio and Iakovidis, Isidoros and Scoccimarro, Enrico},
  title={{sw-kmeanspp} v1.0: spatially weighted k-means++ for the regionalization of climate data (code)},
  publisher={Zenodo},
  howpublished={Zenodo},
  year={2026},
  version={v1.0},
  doi={10.5281/zenodo.XXXXXXX}
}
```

See also `CITATION.cff`. Both have `TODO` placeholders for the Zenodo DOI and
author ORCIDs — fill these in once the first GitHub release mints a DOI (see
"Publishing a release" below).

## Installation

```bash
conda env create -f environment.yml
conda activate swkmeanspp
pip install -e .
```

Requires Python >= 3.10. See `environment.yml` for pinned dependency
versions (resolved against the environment actually used to develop and
verify this code). No `cdo` installation is required — unlike the
project this was extracted from, all preprocessing here is pure
xarray/numpy.

## Data

ERA5 "post-processed daily statistics on single levels from 1940 to
present" (Copernicus Climate Data Store, DOI
[10.24381/cds.4991cf48](https://doi.org/10.24381/cds.4991cf48)), variables
TP, TMAX, SST, MSLP, U10, V10, period 1991–2020.

1. Get a CDS API key and save it to `~/.cdsapirc`:
   https://cds.climate.copernicus.eu/how-to-api
2. Download the variables you need:
   ```bash
   python data_prep/download_era5_cdsapi.py --variable tp tmax sst mslp u10 v10 --start 1991 --end 2020
   ```
   Downloads one file per year per variable under `./data/raw/<variable>/`
   (configurable via `config/paths.yaml`, `--data-root`, or the
   `SWKMEANSPP_DATA_ROOT` environment variable — no machine-specific paths
   anywhere in this repo). `./data/` is gitignored.
3. Preprocessing (climatology, anomaly, domain mask) runs automatically the
   first time `experiments/run_experiment.py` needs it, and is cached under
   `./data/processed/`. To run it standalone:
   ```bash
   python data_prep/preprocess_era5.py --variable tp --domain Europe --start 1991 --end 2020
   ```

ERA5 data files, `~/.cdsapirc`, and all generated output are never committed
to this repository (see `.gitignore`).

## Usage

One YAML config per experiment, one generic CLI entry point — no
hand-edited notebook cells:

```bash
python experiments/run_experiment.py --config config/experiments/tp_europe_sweep.yaml
python experiments/run_experiment.py --config config/experiments/mslp_global.yaml --run no_background
```

Outputs (cluster-label CSVs, centroid CSVs, cluster-diagnostics CSVs, and
map PNGs) are written under `./output/<experiment_name>/`.

## Reproducing the manuscript's results

| Manuscript item | Config | K | `w_s` |
|---|---|---|---|
| Fig. 2, Fig. 4 (TP, Europe, spatial-weight sweep) | `config/experiments/tp_europe_sweep.yaml` | 5 | 2%, 5%, 50%, 100% (+ 5% no-background variant) |
| Fig. 6 (SST, North Atlantic) | `config/experiments/sst_north_atlantic.yaml` | 5 | 5% |
| Fig. 7 (MSLP, Global, with/without background cluster) | `config/experiments/mslp_global.yaml` | 5 | 5%* |
| Fig. 8 (joint U10+V10, Mediterranean) | `config/experiments/wind_mediterranean_joint.yaml` | 5 | 5% |
| Fig. 9, Appendix A (TMAX, Europe) | `config/experiments/tmax_europe.yaml` | 5 | 5%* |

\* Consistent with the rest of the study (Sect. 3.1).

`K=5` is used in every case study, but can be changed by the user.


### Experimental, climatology weight

`experimental.enable_climatology_term` (off by default, and never set in any
of the 5 configs above) is a stub extensibility hook for an experimental
climatology-weighting term — see `swkmeanspp/climatology_experimental.py`.
Setting `beta=0` (the only value used anywhere in this repo) reproduces the
published SW-k-means++ method exactly. 


## Repository structure

```
sw-kmeanspp/
├── swkmeanspp/        # the importable package (algorithm, metrics, plotting)
├── data_prep/          # ERA5 download (cdsapi) and preprocessing
├── config/             # domains.yaml, paths.yaml, one YAML per experiment
├── experiments/        # run_experiment.py, the generic driver
├── tests/
└── docs/method_notes.md
```

## License

MIT — see `LICENSE`.

## Funding

European Union, Horizon 2020 project CLINT (grant agreement No. 101003876),
Horizon Europe project MedEWSa (grant agreement No. 101121192), and Horizon
Europe project SCEWERO (grant agreement No. 101159497).

## Contact

Antonello Squintu — antonello.squintu@cmcc.it

## Publishing a release (for maintainers)

Not performed as part of building this repository — do this yourself, or
ask explicitly for it to be done:

1. Create the GitHub repository and push this code (with `LICENSE`,
   `CITATION.cff`, `README.md`, `environment.yml`, etc. already in place).
2. Log in to [zenodo.org](https://zenodo.org) with "Log in with GitHub"
   (one-time OAuth link between the two accounts).
3. On Zenodo's GitHub settings page
   (zenodo.org/account/settings/github/), find this repository and switch
   its toggle **on** — *before* cutting a release. Zenodo only archives
   releases made after the toggle is enabled.
4. On GitHub, create a **Release** (not just a tag), e.g. tag `v1.0`. This
   triggers Zenodo to pull a snapshot of the repo at that tag and mint a
   DOI automatically.
5. Zenodo mints two DOIs: a **version-specific DOI** (unique to `v1.0`,
   resolves to this exact snapshot) and a **concept DOI** (always resolves
   to the latest version). Use the **version-specific DOI** in the
   manuscript's Code/Data Availability statement, `CITATION.cff`,
   `.zenodo.json`, and this README's citation section — that's what lets a
   reviewer access the exact code used for the paper.
6. Fill in the real DOI (and any ORCIDs) everywhere currently marked `TODO`.
