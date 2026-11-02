# Data

## Covariance matrices

`covariance_matrices_gc.h5` -- **included in this repository** (12 MB)

This HDF5 file contains pre-computed Gaussian-copula covariance matrices derived from the raw time series published in:
- **MA dataset** (Uhrig et al. 2018): https://zenodo.org/records/10572216 (file: `CoCoMac/timeseries.npy`)
- **DBS dataset** (Tasserie et al. 2022): contact corresponding authors

See `notebooks/00_preprocessing.ipynb` for the full preprocessing pipeline.
The notebook regenerates the primary file from the cleaned MA/DBS MATLAB inputs used by the
diagnostics, checking state counts and scan indices against `scan_metadata.csv`. It fits regional
OLS global-signal regression to each full scan in both datasets, then retains the first 250 MA
samples (~600 s) or every other DBS sample (250 points, ~625 s) before the Gaussian copula.
The acquisition band-pass and MA-only notch are unchanged; no-GSR and added-DBS-notch
sensitivities remain in `notebooks/diagnostics/01_B_omega_sensitivity.ipynb`.

## Scan metadata

`scan_metadata.csv` contains one row per scan represented in `covariance_matrices_gc.h5`, with
dataset, state, zero-based scan index within that state, animal, session, and run. The MA mapping
comes from `results/diagnostics/01_A_ma_scan_monkey_map.csv`; the DBS mapping comes from
`metadata_DBS.ods`. Deleted DBS scans are excluded. The 5V-control cleaned index correction (19 to
18) is reflected in the CSV.

HDF5 file with structure:

```
covs/
  MA/
    MA_awake          -> ndarray(n_subjects, 82, 82)
    ts_selv2          -> ndarray(n_subjects, 82, 82)
    ts_selv4          -> ndarray(n_subjects, 82, 82)
    moderate_propofol -> ndarray(n_subjects, 82, 82)
    deep_propofol     -> ndarray(n_subjects, 82, 82)
    ketamine          -> ndarray(n_subjects, 82, 82)
  DBS/
    DBS_awake         -> ndarray(n_subjects, 82, 82)
    ts_on_5V          -> ndarray(n_subjects, 82, 82)
    ts_on_3V          -> ndarray(n_subjects, 82, 82)
    ts_on_3V_control  -> ndarray(n_subjects, 82, 82)
    ts_on_5V_control  -> ndarray(n_subjects, 82, 82)
    ts_off            -> ndarray(n_subjects, 82, 82)
```

The 82 regions are **CoCoMac** parcellation regions.
See `notebooks/00_preprocessing.ipynb` for how to compute these matrices
from raw time series downloaded from Zenodo:
https://zenodo.org/records/10572216  (file: `CoCoMac/timeseries.npy`)
