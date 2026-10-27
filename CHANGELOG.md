# Changelog

## 2026-10-27 — DOI minted

The `v1.0.1` release archived successfully on Zenodo:
DOI `10.5281/zenodo.23697107`. Filled into `CITATION.cff` and `README.md`
(citation BibTeX and version bumped to v1.0.1).

## v1.0.1 — 2026-10-27 — fix invalid Zenodo metadata

The v1.0.0 GitHub release failed to archive on Zenodo. Cause: placeholder
ORCIDs (`0000-0000-0000-0000`) fail the ORCID checksum, and a placeholder
paper DOI (`10.0000/TODO-PAPER-DOI`) was not a real, resolvable identifier —
both likely broke Zenodo's metadata validation. Removed both from
`.zenodo.json`/`CITATION.cff` (to be re-added once real values exist)
rather than leaving invalid placeholders in release metadata.

## v1.0.0 — 2026-10-26 — initial release

First release accompanying the SW-k-means++ manuscript submitted to
Geoscientific Model Development.
