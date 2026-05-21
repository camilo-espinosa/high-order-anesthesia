# Data

## Covariance matrices

`covariance_matrices_gc.h5` -- **included in this repository** (12 MB)

This HDF5 file contains pre-computed Gaussian-copula covariance matrices derived from the raw time series published in:
- **MA dataset** (Uhrig et al. 2018): https://zenodo.org/records/10572216 (file: `CoCoMac/timeseries.npy`)
- **DBS dataset** (Tasserie et al. 2022): contact corresponding authors

See `notebooks/00_preprocessing.ipynb` for the full preprocessing pipeline.

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
