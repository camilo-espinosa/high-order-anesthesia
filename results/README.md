# Results

This directory stores intermediate and final results files.
**Files are not tracked by git** (see `.gitignore`).

## File inventory

| File | Produced by | Consumed by |
|------|------------|-------------|
| `R1_A_max_O_diff_{MA,DBS}_{3..9}.csv` | `R1_A_lopo.ipynb` (Stage 1) | `R1_A_lopo.ipynb` (Stage 2) |
| `R1_A_max_O_diff_{MA,DBS}_all_orders.csv` | `R1_A_lopo.ipynb` (merge) | — |
| `R1_B_nplet_eval_{MA,DBS}.csv` | `R1_A_lopo.ipynb` (Stage 2) | `R2_A_fig3.ipynb`, `S3_order_robustness.ipynb` |
| `R1_C_nplet_tails_PRAUC_with_deltaO_ALL.pkl.gz` | `R1_B_region_sampling.ipynb` | `R1_B_region_sampling.ipynb` (maps step) |
| `R1_C_region_maps_PRAUC_deltaO.pkl.gz` | `R1_B_region_sampling.ipynb` | `R1_C_fig2.ipynb`, `S1_brainmaps_all_orders.ipynb`, `S2_merged_maps.ipynb` |

## Legacy files (safe to delete)

The following files are leftover from earlier runs and are **not used** by the current notebooks:

| File | Notes |
|------|-------|
| `covariance_matrices_gc.h5` | Duplicate — canonical copy is `data/covariance_matrices_gc.h5` |
| `B_2_nplet_tails_PRAUC_with_deltaO_ALL.pkl.gz` | Old name for `R1_C_nplet_tails_PRAUC_with_deltaO_ALL.pkl.gz` |
| `nplet_tails_PRAUC_DBS_3.pkl.gz` | Old intermediate per-dataset/order file |
| `nplet_tails_PRAUC_MA_3.pkl.gz` | Old intermediate per-dataset/order file |

---

## Column conventions

- **`task` / `polarity`** column:
  - `c_gt_nr` → $\Omega_C > \Omega_{NR}$ (conscious-dominated)
  - `nr_gt_c` → $\Omega_{NR} > \Omega_C$ (non-responsive-dominated)
- **`measure`** column: `O` = O-information (Ω), `FC_mean_z` = mean pairwise FC (Fisher z)
- **`PR_AUC`**: PR-AUC with the polarity's positive class as the numerator
- **`PR_AUC_inv`**: PR-AUC with the opposite class as positive
