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
| [`diagnostics/02_search_benchmark.py`](diagnostics/02_search_benchmark.py) | Training-only order-5 exhaustive-versus-annealing comparison for Omega and FC on the same LOAO folds. |

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
| [`00_preprocessing.ipynb`](00_preprocessing.ipynb) | Regenerates the primary covariance file using regional GSR on DBS only, leaving MA unchanged, and duration matching. Required before the new LOAO analysis; the bundled file is not automatically updated. |

---

## R1 — Discrimination of responsive vs non-responsive states

| Notebook |    Description    | Output |
|----------|-------------|--------|
| [`R1_discrimination/R1_A_loao.ipynb`](R1_discrimination/R1_A_loao.ipynb) | **Primary analysis** - resumable ordinary LOAO, animal-balanced training ROC-AUC selection, Omega and symmetric scalar/vector FC baselines, training-only calibration, pooled out-of-fold ROC-/PR-AUC. Raw Omega (O), normalized Omega (O/S) and FC are selected with the same search; a final section refits each selection's own probability models (O/O/S selections: Omega scalar, (TC, DTC) vector and first-order Omega-gradient vector; FC selections: mean FC scalar and FC vector; families never crossed) and reports ROC-AUC/PR-AUC. Scalar pairwise versus scalar Omega is the dimension-matched comparison; pairwise vectors require a matched-dimensional HOI vector for a fair comparison. | `results/R1_A_loao/{dataset}_{predictions,selections,metrics,stability,region_frequency,sessions,fits}.csv`; feature-set evaluation: `results/R1_A_loao/{dataset}_features_{predictions,fits,pooled,animal,animal_summary}.csv` |
| [`R1_discrimination/R1_D_synergy_loao.ipynb`](R1_discrimination/R1_D_synergy_loao.ipynb) | Synergy-constrained LOAO for raw Omega (`Omega_R` below zero in training); evaluates direction-enforced Omega, matched-region mean FC/vector, their combined model, and the best mean-FC reference anywhere. Reports pooled ROC-AUC/PR-AUC and fold-level AUCs when both classes are present; single-class folds are non-evaluable. | `results/R1_D_synergy/{dataset}_{selections,predictions,fits,pooled,animal,animal_summary}.csv`; additionally `{dataset}_fold_metrics.csv` |
| [`R1_discrimination/R1_X_exhaustive_maximum.ipynb`](R1_discrimination/R1_X_exhaustive_maximum.ipynb) | Exploratory exhaustive search across all animals within each dataset. Retains the top K order-3/4 n-plets for Omega, mean Fisher-z, and a same-scan pairwise-vector logistic model in both score directions; stores all three feature representations and their AUCs for cross-evaluation. The apparent maxima are selection-biased and not generalization estimates. | `results/R1_X_exhaustive_maximum/{dataset}_{top_candidates,selected_candidate_summary,selected_scan_values}.csv` |
| [`supplementary/S4_lopo_(old implementation).ipynb`](supplementary/S4_lopo_(old%20implementation).ipynb) | Archived scan-pair selection/LOPO implementation. Not the primary generalization estimate. | Legacy pair-based CSVs |
| [`R1_discrimination/R1_B_region_sampling.ipynb`](R1_discrimination/R1_B_region_sampling.ipynb) | Sample millions of random n-plets, score each with PR-AUC, retain the top percentile, and aggregate into per-region participation maps (ΔΩ-weighted). | `results/R1_C_nplet_tails_PRAUC_with_deltaO_ALL.pkl.gz`, `results/R1_C_region_maps_PRAUC_deltaO.pkl.gz` |
| [`R1_discrimination/R1_C_fig2.ipynb`](R1_discrimination/R1_C_fig2.ipynb) | **Figure 2** – Violin plots of Ω per condition for the optimal n-plets (MWU + Bonferroni correction) and CoCoMac brain maps of regional participation. | Fig 2 |

---

### Primary LOAO workflow

The assistant prepares changes and static checks; the user runs scripts, notebooks and tests.

1. Run [00](00_preprocessing.ipynb) to regenerate the GSR/duration-matched HDF5.
2. Before the main analysis, run the separate order-5 benchmark from the repository root:

   ```powershell
   python notebooks\diagnostics\02_search_benchmark.py
   ```

   Optional `--datasets MA --animals A` restricts the benchmark. It never evaluates held-out scores.
   The exhaustive reference covers 27,285,336 candidates per fold with 82 regions; assess runtime
   before launching all folds. Timing columns cover both polarities together, not each direction.
   The benchmark is diagnostic, not an automatic method selector: order 5 remains exhaustive.
3. Optionally run the **GPU smoke pass** cell in [R1_A_loao](R1_discrimination/R1_A_loao.ipynb).
   It scores one 2,048-candidate batch per exhaustive order and one SA iteration per restart;
   it does not evaluate held-out scans or save results. The smoke cell and both LOAO runners
   release temporary CUDA tensors and clear unused allocator cache; tensors retained elsewhere
   in the notebook kernel cannot be freed by these calls.
4. Run the separate exhaustive, annealing, and save cells in R1_A_loao for the full analysis.
   Each completed dataset/held-out-animal/order/measure unit is checkpointed atomically under
   `results/R1_A_loao/checkpoints/<run-id>/`; `run.log` records starts, completions, checkpoint reuse,
   and full exception tracebacks. The run ID fingerprints the covariance file, metadata, datasets,
   device, search settings, and order configuration. After an interruption, rerun the setup, settings,
   inputs, and affected search cells: matching units are reused, while changed inputs/settings get a
   separate run directory. After an interruption, restart the kernel before rerunning setup,
   settings, inputs, and the affected search cell. Run the save and summary cells after both
   search cells complete.
   Exhaustive search displays candidate-level progress bars; annealing displays iteration progress
   for each measure/polarity.

Fixed settings: exhaustive candidates are scored in batches of 2,048; simulated annealing uses
100 restarts, 10,000 iterations, 1,000 iterations without a strict AUC improvement before early
stopping, temperature 0.05, cooling 0.99 and base seed 2026.
The budget follows archived LOPO; the AUC-scale temperature was approved separately.
Orders 3-5 are exhaustive, orders 6-9 use the existing shared THOI-derived annealer with
candidate-score callbacks. FC never calls the HOI estimator. Both polarities use independent
searches, with exact objective ties resolved by the lexicographically smallest sorted region tuple
among evaluated candidates. Plateau tie replacements do not reset early stopping.

Training weights are `1/(A_c * n_a,c)`, recomputed per fold. Each class has total weight one.
Weighted scaling and fixed L2 logistic fits (`C=1`, weights total two, no inner CV) use training scans
only. Scalars use the training-selected polarity; vector features are upper-triangle Fisher-z
correlations in sorted selected-region order. Omega uses THOI with `T=None`, matching the existing
covariance evaluation convention. Primary inputs have equal sample counts across scans.

Metrics pool only responsive probabilities, never raw Omega across folds. PR-AUC means weighted
average precision; chance is the saved weighted prevalence (0.5), not the raw scan fraction.
Predictions retain animal/state/scan/session/run, held-out animal, selected n-plet, raw values and
probabilities. The fit table records training means/scales, coefficients and intercepts for reuse;
array-valued CSV cells are JSON. Jaccard overlap, region frequency and session counts are exported separately.

These are estimates of a selection procedure, not validation of a universal n-plet. DBS animal/state
confounding and few animals remain limitations. Mixed models and whole-pipeline permutation tests
are Phase 2, not part of this notebook. Existing figure, brain-map and legacy CSV consumers remain
descriptive/legacy and are not overwritten or automatically converted to LOAO.

Focused checks for the user to run:

```powershell
python -m unittest discover -s tests -p test_validation.py
```

---

### R1_D — Synergy-constrained LOAO follow-up

Run [`R1_D_synergy_loao.ipynb`](R1_discrimination/R1_D_synergy_loao.ipynb) after preprocessing. It performs polarity -1 raw-Omega selection with an animal-weighted training-awake median below zero, plus a best-mean-FC-anywhere reference. Exhaustive orders 3-5 traverse candidates once per order and update a separate best selection for each uncached fold; annealing orders 6-9 remain fold-specific. Results are checkpointed per fold under `results/R1_D_synergy/checkpoints/<run-id>/`; empty admissible folds remain empty and are reported.

The held-out models use the same Omega-selected regions for direction-enforced Omega, mean FC, the FC vector, and FC vector plus Omega; the pairwise reference uses its separately selected n-plet. Criterion 1 uses pooled Omega ROC-AUC above 0.5; criterion 3 treats both matched-region pairwise models as near chance only when their ROC-AUCs are at most 0.55, otherwise the FC-vector-plus-Omega model must improve ROC-AUC over the FC vector. PR-AUC is reported alongside ROC-AUC, with pooled-score scale caveats for fold-specific models. Scalar Omega and scalar mean-FC form the dimension-matched comparison; a pairwise vector is not a fair comparison to scalar Omega without a matched-dimensional HOI vector. Results are saved to `results/R1_D_synergy/{dataset}_{selections,predictions,fits,pooled,animal,animal_summary}.csv` plus per-fold ROC-/PR-AUC in `{dataset}_fold_metrics.csv`. The notebook does not overwrite R1_A outputs.

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
