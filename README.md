# High-order brain interactions distinguish awake states, general anesthesia, and recovery under deep brain stimulation

Code accompanying the paper:

> **High-order brain interactions distinguish awake states, general anesthesia, and recovery under deep brain stimulation**  
> [manuscript](https://doi.org/10.64898/2026.06.01.728390)

---

## Overview

We apply multivariate information theory — specifically O-information (Ω) — to resting-state fMRI data from non-human primates to characterise how higher-order brain interactions reorganise across states of responsiveness. Ω > 0 indicates redundancy-dominated dynamics; Ω < 0 indicates synergy-dominated dynamics.

Two complementary datasets are analysed:

| Dataset | Conditions |
|---------|-----------|
| **MA** (multi-anesthesia) | Wakefulness + propofol, sevoflurane, ketamine |
| **DBS** (deep brain stimulation) | Wakefulness + propofol anesthesia + central-thalamus stimulation at two intensities |

The primary revision analysis uses ordinary leave-one-animal-out (LOAO) validation. Within each
training fold and *n*-plet size *n* ∈ {3, …, 9}, subsets maximise animal-balanced ROC-AUC for
**responsive (C)** versus **non-responsive (NR)** scans under two optimisation polarities:

- **Ω_C > Ω_NR** — redundancy elevated in wakefulness
- **Ω_NR > Ω_C** — synergy-to-redundancy transition in non-responsiveness

Orders 3-5 use exhaustive search; orders 6-9 use simulated annealing. Symmetric baselines optimise
mean within-n-plet Fisher-z FC and fit full pairwise-vector models on both selected sets.
Training-only logistic calibration allows held-out probabilities to be pooled for animal-balanced
ROC-AUC and PR-AUC (weighted average precision). The archived LOPO pipeline and existing
figures/results remain legacy outputs until explicitly regenerated; no new performance is claimed
before the user runs the revision analysis.

For MA, classification compares awake scans against deep propofol, ketamine, and
deep sevoflurane (24 reference and 56 NR scans). Moderate propofol and moderate
sevoflurane are retained in condition-wise plots but excluded from optimisation,
scores, regional-map calculations, and primary statistical comparisons. DBS
classification is unchanged; low CT remains descriptive only.

---

## Repository structure

```
high-order-anesthesia/
├── data/
│   ├── covariance_matrices_gc.h5   # Pre-computed matrices
│   └── README.md                   # HDF5 format documentation
├── results/
│   └── README.md                   # Intermediate file inventory
├── src/hoi_anesthesia/             # Shared Python utilities
│   ├── io.py                       # load_covariance_dict, save_results
│   ├── utils.py                    # max_difference_pairs, evaluate_nplet_batched, …
│   ├── thoi_utils.py               # simulated_annealing_parallel wrapper
│   ├── validation.py               # Animal-balanced LOAO selection and baselines
│   ├── preprocessing.py
│   ├── stats.py
│   └── plotting.py                 # plot_cocomac_region_values (Plotly brain maps)
└── notebooks/
    ├── 00_preprocessing.ipynb           # Covariance matrix computation (example)
    ├── R1_discrimination/
    │   ├── R1_A_loao.ipynb              # Primary animal-held-out validation and FC baselines
    │   ├── R1_D_synergy_loao.ipynb      # Synergy-constrained matched-region pairwise test
    │   ├── R1_B_region_sampling.ipynb   # Region participation maps
    │   └── R1_C_fig2.ipynb              # Violin plots + brain maps                           → Fig 2
    ├── R2_order_effects/
    │   └── R2_A_fig3.ipynb              # Boxplots + PR-AUC vs order                          → Fig 3
    ├── R3_cross_dataset/
    │   └── R3_A_fig4.ipynb              # Cross-dataset evaluation + Table 1                  → Fig 4
    └── supplementary/
        ├── S1_brainmaps_all_orders.ipynb   # Brain maps at all orders 3–9                     → Fig S5
        ├── S2_merged_maps.ipynb            # Merged MA+DBS participation maps                 → Fig S6
        ├── S3_order_robustness.ipynb       # Top-50 n-plets robustness check                  → Fig S3
        └── S4_combined_dataset.ipynb       # Optimal n-plets on combined MA+DBS               → Fig S4b
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/camilo-espinosa/high-order-anesthesia.git
cd high-order-anesthesia
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .high_order_anesthesia_repo
# Windows
.high_order_anesthesia_repo\Scripts\activate
# macOS / Linux
source .high_order_anesthesia_repo/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA support (recommended for GPU acceleration):

```bash
# CUDA 12.4 example — adjust to your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for other CUDA versions.

---

## Data

The covariance matrix file `data/covariance_matrices_gc.h5` is **included in this repository** (12 MB).

These matrices were computed from raw fMRI time series published with the original datasets:

- **MA dataset**: Uhrig et al. (2018) — [https://zenodo.org/records/10572216](https://zenodo.org/records/10572216) (`CoCoMac/timeseries.npy`)
- **DBS dataset**: Tasserie et al. (2022) — contact corresponding authors

The notebook `notebooks/00_preprocessing.ipynb` documents the full preprocessing pipeline for reproducibility. See `data/README.md` for the HDF5 schema.

---

## Running the notebooks

Run notebooks in order from the project root (`high-order-anesthesia/`). Each notebook sets `os.chdir` automatically if run from a subdirectory.

| Step | Notebook | Output |
|------|----------|--------|
| 0 | `00_preprocessing.ipynb` | `data/covariance_matrices_gc.h5` |
| 1 | [`R1_A_loao.ipynb`](notebooks/R1_discrimination/R1_A_loao.ipynb) | `results/R1_A_loao/` out-of-fold predictions, selections, metrics, stability and model fits |
| 1a | [`R1_D_synergy_loao.ipynb`](notebooks/R1_discrimination/R1_D_synergy_loao.ipynb) | `results/R1_D_synergy/` synergy-constrained selections and matched-region held-out model tables |
| 2 | `R1_B_region_sampling.ipynb` | `results/R1_C_nplet_tails_*.pkl.gz`, `results/R1_C_region_maps_*.pkl.gz` |
| 3 | `R1_C_fig2.ipynb` | Fig 2 (violin plots + brain maps) |
| 4 | `R2_A_fig3.ipynb` | Fig 3 (order effects + PR-AUC lines) |
| 5 | `R3_A_fig4.ipynb` | Fig 4 + Table 1 (cross-dataset) |
| S1–S4 | `supplementary/` | Supplementary figures |

Steps 0–2 and 1a are computationally intensive (GPU recommended for steps 1 and 1a).
Step 1a is a separate, optional R1 follow-up; criterion 3 defines near chance as both same-region pairwise models having animal-balanced accuracy at or below 0.55.
Steps 3–5 and supplementary notebooks only load pre-computed results.
They still consume legacy selections; they do not automatically consume the new LOAO outputs.
Regenerate primary covariances with step 0, then assess the separate order-5 training-fold benchmark
before running step 1. See the [LOAO workflow](notebooks/README.md#primary-loao-workflow) for fixed
settings, output schemas and commands. Scripts, notebook cells and tests are run by the user.

---

## Citation

If you use this code, please cite the associated paper (citation details will be updated upon publication) and the THOI library:

```bibtex
@article{belloli2025thoi,
  title={THOI: An efficient and accessible library for computing higher-order interactions enhanced by batch-processing},
  author={Belloli, Laouen and Mediano, Pedro and Cofre, Rodrigo and Slezak, Diego Fernandez and Herzog, Ruben},
  journal={PLoS One},
  volume={21},
  number={5},
  pages={e0348005},
  year={2026}
}
```
