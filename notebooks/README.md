# Notebooks

All notebooks should be run from the repository root (`high-order-anesthesia/`).
Each notebook sets `sys.path` and `os.chdir` automatically.

Run notebooks in the order below. Computationally intensive steps benefit from a CUDA GPU via the THOI library.

---

## Diagnostics - preprocessing sensitivity

| Notebook | Tests |
|----------|-------|
| [`diagnostics/01_A_preprocessing_assumptions.ipynb`](diagnostics/01_A_preprocessing_assumptions.ipynb) | Regional global signal, spectra/notch, effective sample size and segment covariance reliability. |
| [`diagnostics/01_B_omega_sensitivity.ipynb`](diagnostics/01_B_omega_sensitivity.ipynb) | Exhaustive orders 3 and 4 only for V0/V1/V2/V3/V4/V3r; per-dataset and pooled separation scores; four figure cells per version. |

Each test has a short description, plots and an interpretation from the previous diagnostic run.
The new exhaustive results are not anticipated in the interpretation cells.
Phase 2 defines its variants directly; it does not depend on Phase 1 flags.
Outputs in `results/diagnostics/` include figures, covariance variants, CSV summaries and
`01_B_nplet_omega_summaries.h5` (every scan-level Omega in float64, class medians, mean C-minus-NR
delta and `pr_auc_C` / `pr_auc_NR`). Within each `variant/dataset/order` group,
`omega[scan, nplet]` aligns with zero-based region indices in `nplets[nplet, region]`
and dataset/state/class/local scan indices in `scans/dataset`. Region indices are shared hard links
to `nplets/order`.
All states, including descriptive states, are saved; only C/NR scans enter class summaries.
The `combined` groups contain scores calculated directly from pooled MA/DBS scans during each batch,
not averages of dataset scores; combined scan-level values remain stored in MA and DBS without duplication.
Both APs use higher Omega as the score, with C or NR as positive; ties match scikit-learn average precision.
The maximum of each AP selects one n-plet per positive class, not the maximum/minimum of a single AP.
Exhaustive enumeration writes batches directly to compressed HDF5 (about 24 GB before compression
for all six variants). Rerunning the calculation cell recreates this file.
Each order clears unused GPU cache in `finally` and reports PyTorch live/reserved memory;
MA/DBS share a 512-n-plet batch so pooled scores do not require a second pass over saved Omega.
Cache cleanup cannot free live tensors retained by interrupted tracebacks; restart that kernel
if live memory stays high. These edits take effect on the next execution, not in an already running cell.
Each version has four separate figure cells: all-state Omega violins and C/NR violins for
the C-positive and NR-positive AP maxima selected on MA, DBS and pooled scans.
Each figure includes orders 3/4 as panels. All-state violins use every scan/n-plet value in a
chunked 1,024-bin Gaussian density estimate, with no subsampling; selected-n-plet plots use every
relevant C/NR scan. The selections and figures are saved when their plot cells run.
The plotting setup and individual plot cells can run from the project directory without recalculation.
The final summary compares maximum saved PR-AUC across preprocessing versions separately for
MA, DBS and pooled scans, each order and each positive class, retaining exact ties.
It saves all version maxima and the winning versions to `01_B_preprocessing_pr_auc_comparison.csv`
and `01_B_best_preprocessing_pr_auc.csv`. These are selected in-sample scores.
Legacy reports, verdict JSONs, random-sampling exports and superseded figures are not refreshed.

---

## 00 — Preprocessing (optional)

| Notebook | Description |
|----------|-------------|
| [`00_preprocessing.ipynb`](00_preprocessing.ipynb) | Documents how per-scan Gaussian-copula covariance matrices were computed from raw fMRI time series. **As example** — matrices used in the study are already included in `data/covariance_matrices_gc.h5`. Use this notebook only if you want to recompute from raw data. |

---

## R1 — Discrimination of responsive vs non-responsive states

| Notebook |    Description    | Output |
|----------|-------------|--------|
| [`R1_discrimination/R1_A_lopo.ipynb`](R1_discrimination/R1_A_lopo.ipynb) | **Stage 1** – Simulated annealing finds the *n*-plet (per dataset, order 3–9, polarity) that maximises ΔΩ across C/NR scan pairs. **Stage 2** – Leave-one-pair-out (LOPO) PR-AUC evaluation of discovered n-plets. | `results/R1_A_max_O_diff_{dataset}_{order}.csv`, `results/R1_A_max_O_diff_{dataset}_all_orders.csv`, `results/R1_B_nplet_eval_{dataset}.csv` |
| [`R1_discrimination/R1_B_region_sampling.ipynb`](R1_discrimination/R1_B_region_sampling.ipynb) | Sample millions of random n-plets, score each with PR-AUC, retain the top percentile, and aggregate into per-region participation maps (ΔΩ-weighted). | `results/R1_C_nplet_tails_PRAUC_with_deltaO_ALL.pkl.gz`, `results/R1_C_region_maps_PRAUC_deltaO.pkl.gz` |
| [`R1_discrimination/R1_C_fig2.ipynb`](R1_discrimination/R1_C_fig2.ipynb) | **Figure 2** – Violin plots of Ω per condition for the optimal n-plets (MWU + Bonferroni correction) and CoCoMac brain maps of regional participation. | Fig 2 |

---

## R2 — Order effects

| Notebook | Description | Output |
|----------|-------------|--------|
| [`R2_order_effects/R2_A_fig3.ipynb`](R2_order_effects/R2_A_fig3.ipynb) | **Figure 3** – Ω boxplots per interaction order (C vs NR macrostates) and PR-AUC vs order line plots comparing HOI against mean functional connectivity (FC_mean_z). | Fig 3 |

---

## R3 — Cross-dataset generalisation

| Notebook | Description | Output |
|----------|-------------|--------|
| [`R3_cross_dataset/R3_A_fig4.ipynb`](R3_cross_dataset/R3_A_fig4.ipynb) | **Figure 4** – Evaluates each dataset's optimal n-plets on the *opposite* dataset (cross) and same dataset (within). Violin plots + **Table 1** (PR-AUC summary, LaTeX output). | Fig 4, Table 1 |

---

## Supplementary

| Notebook | Description | Figure |
|----------|-------------|--------|
| [`supplementary/S1_brainmaps_all_orders.ipynb`](supplementary/S1_brainmaps_all_orders.ipynb) | CoCoMac brain maps for every interaction order 3–9 (both polarities, both datasets). | Fig S5 |
| [`supplementary/S2_merged_maps.ipynb`](supplementary/S2_merged_maps.ipynb) | Merged MA+DBS regional participation maps. | Fig S6 |
| [`supplementary/S3_order_robustness.ipynb`](supplementary/S3_order_robustness.ipynb) | Repeats `R2_A_fig3` using the **top-50** n-plets instead of the single best, to verify robustness of order effects. | Fig S3 |
| [`supplementary/S4_combined_dataset.ipynb`](supplementary/S4_combined_dataset.ipynb) | Violin plots for all four optimal n-plets evaluated on the combined MA+DBS scan pool. | Fig S4b |

---
