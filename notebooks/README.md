# Notebooks

All notebooks should be run from the repository root (`high-order-anesthesia/`).
Each notebook sets `sys.path` and `os.chdir` automatically.

Run notebooks in the order below. Computationally intensive steps benefit from a CUDA GPU via the THOI library.

---

## 00 — Preprocessing (optional)

| Notebook | Description |
|----------|-------------|
| [`00_preprocessing.ipynb`](00_preprocessing.ipynb) | Documents how per-scan Gaussian-copula covariance matrices were computed from raw fMRI time series. **As example** — matrices used in the study are already included in `data/covariance_matrices_gc.h5`. Use this notebook only if you want to recompute from raw data. |

---

## R1 — Discrimination of conscious vs non-responsive states

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

