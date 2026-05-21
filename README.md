# High-order brain interactions distinguish awake states, general anesthesia, and recovery under deep brain stimulation

Code accompanying the paper:

> **High-order brain interactions distinguish awake states, general anesthesia, and recovery under deep brain stimulation**  
> *(manuscript in preparation)*

---

## Overview

We apply multivariate information theory — specifically O-information (Ω) — to resting-state fMRI data from non-human primates to characterise how higher-order brain interactions reorganise across states of consciousness. Ω > 0 indicates redundancy-dominated dynamics; Ω < 0 indicates synergy-dominated dynamics.

Two complementary datasets are analysed:

| Dataset | Conditions |
|---------|-----------|
| **MA** (multi-anesthesia) | Wakefulness + propofol, sevoflurane, ketamine |
| **DBS** (deep brain stimulation) | Wakefulness + propofol anesthesia + central-thalamus stimulation at two intensities |

For each dataset and each *n*-plet size *n* ∈ {3, …, 9}, simulated annealing identifies regional subsets whose Ω best separates **conscious (C)** from **non-responsive (NR)** scans under two optimisation polarities:

- **Ω_C > Ω_NR** — redundancy elevated in wakefulness
- **Ω_NR > Ω_C** — synergy-to-redundancy transition in non-responsiveness

Discrimination is quantified by PR-AUC in a leave-one-pair-out (LOPO) cross-validation scheme.

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
│   ├── preprocessing.py
│   ├── stats.py
│   └── plotting.py                 # plot_cocomac_region_values (Plotly brain maps)
└── notebooks/
    ├── 00_preprocessing.ipynb           # Covariance matrix computation (example)
    ├── R1_discrimination/
    │   ├── R1_A_lopo.ipynb              # Stage 1: simulated annealing; Stage 2: LOPO PR-AUC  
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
| 1 | `R1_A_lopo.ipynb` | `results/R1_A_max_O_diff_*.csv`, `results/R1_B_nplet_eval_*.csv` |
| 2 | `R1_B_region_sampling.ipynb` | `results/R1_C_nplet_tails_*.pkl.gz`, `results/R1_C_region_maps_*.pkl.gz` |
| 3 | `R1_C_fig2.ipynb` | Fig 2 (violin plots + brain maps) |
| 4 | `R2_A_fig3.ipynb` | Fig 3 (order effects + PR-AUC lines) |
| 5 | `R3_A_fig4.ipynb` | Fig 4 + Table 1 (cross-dataset) |
| S1–S4 | `supplementary/` | Supplementary figures |

Steps 0–2 are computationally intensive (GPU recommended for step 1).
Steps 3–5 and supplementary notebooks only load pre-computed results.

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
