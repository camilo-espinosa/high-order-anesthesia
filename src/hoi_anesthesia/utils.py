from sklearn.metrics import average_precision_score
from sklearn.feature_selection import f_classif
import numpy as np
import torch
import logging
from thoi.measures.gaussian_copula import nplets_measures
from typing import List, Tuple
import logging
import heapq
from tqdm import tqdm
import itertools
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import f_classif
from thoi.measures.gaussian_copula import nplets_measures
import math


def max_difference_pairs(batched_measures, measure="o"):
    """
    batched_measures: (batch, D, 4), with D = 2*N_pairs
    returns:  (batch, D) -- one score per dataset entry (duplicated per pair)
    """
    B, D, _ = batched_measures.shape
    assert D % 2 == 0
    measure_dict = {"tc": 0, "dtc": 1, "o": 2, "s": 3}
    measure_idx = measure_dict[measure]
    O = batched_measures[..., measure_idx]  # (B, D)
    diffs = O[..., 0] - O[..., 1]  # (B, N_pairs)
    return diffs


def evaluate_nplet_batched(
    idx,
    all_covs,
    conscious_states,
    nonresponsive_states,
    selected_dataset,
    optimal_nplet,
    state_c,  # tuple-like (ds, st) for discovery in conscious set
    state_nr,  # tuple-like (ds, st) for discovery in nonresponsive set
    subject_c,  # index to skip in conscious discovery pair
    subject_nr,  # index to skip in nonresponsive discovery pair
    device,
):
    """
    Batched evaluation of a single n-plet:
      - Computes TC, DTC, O, S, norm_O, FC_mean_z in one pass.
      (FC_mean_Z: mean Fisher-z correlation within n-plet)
      - Processes all subjects per condition in a single call to nplets_measures
        by passing a list of covariance matrices (covmat_precomputed=True).

    Returns
    -------
    list of dicts, one per measure, with:
        row_idx, measure, F_score, PR_AUC, PR_AUC_inv
    """
    # HOI measures + classical FC
    measure_list = ["TC", "DTC", "O", "S", "norm_O", "FC_mean_z"]

    # --- Gather covariance matrices per condition (skip discovery subject)
    covs_conscious = []
    for st in conscious_states[selected_dataset]:
        covs = all_covs[selected_dataset][st]  # (n_subj, N, N)
        for subj_idx in range(covs.shape[0]):
            if st == state_c and subj_idx == subject_c:
                continue
            covs_conscious.append(np.asarray(covs[subj_idx]))

    covs_nonresp = []
    for st in nonresponsive_states[selected_dataset]:
        covs = all_covs[selected_dataset][st]
        for subj_idx in range(covs.shape[0]):
            if st == state_nr and subj_idx == subject_nr:
                continue
            covs_nonresp.append(np.asarray(covs[subj_idx]))

    n_c = len(covs_conscious)
    n_nr = len(covs_nonresp)

    if n_c == 0 or n_nr == 0:
        raise ValueError(
            "Empty group after skipping discovery subject(s). Check inputs."
        )

    # --- Single batched tensor over ALL subjects (conscious first, then nonresponsive)
    X_list = covs_conscious + covs_nonresp  # list of (N,N)
    X_array = np.array(X_list)  # (D, N, N)
    # keep dtype from numpy; move to device for GPU if requested
    X_tensor = torch.as_tensor(X_array, device=device)

    # --- HOI measures via THOI (TC, DTC, O, S)
    measures = nplets_measures(
        X_tensor,
        nplets=[optimal_nplet],  # single n-plet
        covmat_precomputed=True,
        T=None,  # keep same behavior as your original code
        device=device,
        verbose=logging.WARNING,
    )
    # measures shape: (1, D, 4)
    vals_hoi = measures[0, :, :4].detach().cpu().numpy()  # (D, 4)

    # norm_O = O / S
    ratio = vals_hoi[:, 2] / vals_hoi[:, 3]  # shape (D,)

    # --- Classical FC:  (batched, on device)
    with torch.no_grad():
        idx_t = torch.as_tensor(optimal_nplet, dtype=torch.long, device=device)
        # Sub-covariance for n-plet: (D, k, k)
        cov_sub = X_tensor.index_select(1, idx_t).index_select(2, idx_t)

        # Convert covariance to correlation
        # var: (D, k)
        var = torch.diagonal(cov_sub, dim1=-2, dim2=-1)
        eps = 1e-12
        std = torch.sqrt(torch.clamp(var, min=eps))  # (D, k)
        denom = std.unsqueeze(-1) * std.unsqueeze(-2)  # (D, k, k)
        corr = cov_sub / torch.clamp(denom, min=eps)

        # numerical safety for Fisher z
        corr = torch.clamp(corr, -0.999999, 0.999999)

        k = idx_t.numel()
        # upper triangle mask, excluding diagonal
        triu_mask = torch.triu(torch.ones(k, k, dtype=bool, device=device), diagonal=1)
        # (D, K) where K = k(k-1)/2
        corr_pairs = corr[:, triu_mask]
        z_pairs = torch.atanh(corr_pairs)
        fc_mean_z_tensor = z_pairs.mean(dim=-1)  # (D,)

    fc_mean_z = fc_mean_z_tensor.detach().cpu().numpy()  # (D,)

    # --- Stack all features: [TC, DTC, O, S, norm_O, FC_mean_z]
    vals = np.column_stack((vals_hoi, ratio, fc_mean_z))  # (D, 6)

    # --- Labels: 1 for conscious, 0 for nonresponsive
    y = np.concatenate([np.ones(n_c, dtype=int), np.zeros(n_nr, dtype=int)])  # (D,)

    # --- ANOVA F-score across all columns
    F_vals, _ = f_classif(vals, y)  # shape (6,)
    # Guard against zero variance
    for j in range(vals.shape[1]):
        if np.allclose(vals[:, j].var(), 0):
            F_vals[j] = 0.0

    # --- PR AUC & inverse-PR AUC per column
    pr_aucs = []
    neg_pr_aucs = []
    for j in range(vals.shape[1]):
        xj = vals[:, j]
        try:
            pr_aucs.append(average_precision_score(y, xj))
            neg_pr_aucs.append(average_precision_score(y, -xj))
        except ValueError:
            pr_aucs.append(np.nan)
            neg_pr_aucs.append(np.nan)

    out_list = []
    for jdx, measure_ in enumerate(measure_list):
        out_list.append(
            {
                "row_idx": idx,
                "measure": measure_,
                "F_score": F_vals[jdx],
                "PR_AUC": pr_aucs[jdx],
                "PR_AUC_inv": neg_pr_aucs[jdx],
            }
        )

    return out_list


def generate_X(conscious_states, nonresponsive_states, all_covs):
    covs_conscious = []
    covs_conscious_name = []
    for ds, states in conscious_states.items():
        for st in states:
            covs = all_covs[ds][st]
            for subj_idx in range(covs.shape[0]):
                covs_conscious.append(np.asarray(covs[subj_idx]))
                covs_conscious_name.append([ds, st, subj_idx])
    covs_nonresp = []
    covs_nonresp_name = []
    for ds, states in nonresponsive_states.items():
        for st in states:
            covs = all_covs[ds][st]
            for subj_idx in range(covs.shape[0]):
                covs_nonresp.append(np.asarray(covs[subj_idx]))
                covs_nonresp_name.append([ds, st, subj_idx])

    n_c = len(covs_conscious)
    n_nr = len(covs_nonresp)
    X_array = np.array(covs_conscious + covs_nonresp)
    X_tensor = torch.tensor(X_array)
    return X_tensor, n_c, n_nr


def sample_unique_nplets(
    M: int,
    k: int,
    batch_size: int = 100000,
    R: int = 82,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Generate M unique n-plets of size k sampled uniformly at random.
    Returns np.array of shape (M, k) with sorted indices.
    """
    max_possible = math.comb(R, k)

    if M > max_possible:
        print(
            "WARNING: Requested more n-plets than exist. "
            "Returning all possible combinations instead."
        )
        all_combos = list(itertools.combinations(range(R), k))
        return torch.tensor(all_combos, dtype=torch.int16)

    seen = set()
    result = []

    while len(result) < M:
        # generate candidates (using THOI's strategy)
        cand = (
            torch.stack(
                [torch.randperm(R, device=device)[:k] for _ in range(batch_size)]
            )
            .cpu()
            .numpy()
        )

        for row in cand:
            tpl = tuple(sorted(row))
            if tpl not in seen:
                seen.add(tpl)
                result.append(tpl)
                if len(result) >= M:
                    break

    return torch.from_numpy(np.array(result, dtype=np.int16))


def O_Fscore(X, n_c):
    N_subjects = X.shape[0]
    y = np.zeros(N_subjects, dtype=int)
    y[:n_c] = 1  # 1 = C, 0 = NR
    # ANOVA F-score (feature-wise)
    F_vals, _ = f_classif(X, y)
    return F_vals


def O_PR_AUC(X, n_c):
    N_subjects = X.shape[0]
    B = X.shape[1]
    y = np.zeros(N_subjects, dtype=int)
    y[:n_c] = 1  # 1 = C, 0 = NR
    pr_c_pos = np.empty(B, dtype=float)
    pr_nr_pos = np.empty(B, dtype=float)

    # y_NRpos = 1 - y  # 1 = NR, 0 = C
    for i in range(B):
        scores = X[:, i]  # (N,)
        # PR AUC with C as positive
        pr_c_pos[i] = average_precision_score(y, scores)
        # PR AUC with NR as positive
        pr_nr_pos[i] = average_precision_score(y, -scores)
    return pr_c_pos, pr_nr_pos


def delta_O(X, n_c):
    O_C = X[:n_c, :]  # (n_c, B)
    O_NR = X[n_c:, :]  # (N_subjects-n_c, B)
    mean_C = O_C.mean(axis=0)  # (B,)
    mean_NR = O_NR.mean(axis=0)  # (B,)
    delta_O = mean_C - mean_NR  # (B,)
    delta_O_NR = mean_NR - mean_C  # (B,)
    return delta_O, delta_O_NR


@torch.no_grad()
def compute_O_and_delta_batch(
    X_tensor: torch.Tensor,
    nplets_batch: torch.Tensor,
    n_c: int,
    device: torch.device = torch.device("cpu"),
    eval_fc=O_PR_AUC,
):
    """
    Compute O-information for a batch of n-plets over all subjects,
    and the corresponding ΔO = mean(O_C) - mean(O_NR) per n-plet.

    Parameters
    ----------
    X_tensor : torch.Tensor
        Covariance matrices for all subjects, shape (N_subjects, R, R).
        The first n_c subjects are Conscious (C), the remaining are Non-Responsive (NR).
    nplets_batch : torch.Tensor
        Batch of n-plets, shape (B, k), with indices in [0, R-1].
    n_c : int
        Number of conscious subjects. NR subjects are from n_c to N_subjects-1.
    batch_size : int, optional
        Batch size argument passed to nplets_measures.
    device : torch.device, optional
        Device on which to run the computation.

    Returns
    -------
    O_vals : torch.Tensor
        O-information values, shape (N_subjects, B).
        O_vals[s, i] is O for subject s and n-plet i.
    delta_O : torch.Tensor
        ΔO per n-plet, shape (B,).
        delta_O[i] = mean_C(O_vals[:, i]) - mean_NR(O_vals[:, i]).
    """
    X_tensor = X_tensor.to(device=device, dtype=torch.float64)  # (N, R, R)
    nplets_batch = nplets_batch.to(device=device, dtype=torch.long)  # (B, k)
    measures = nplets_measures(
        X=X_tensor,
        covmat_precomputed=True,
        nplets=nplets_batch,
        # batch_size=batch_size,
        device=device,
        verbose=logging.WARNING,
    )
    O_vals = measures[..., 2].detach().cpu().numpy()
    O_vals = O_vals.transpose(1, 0)
    pr_c_pos, pr_nr_pos = eval_fc(O_vals, n_c)
    return pr_c_pos, pr_nr_pos


class TailKeeperPR:
    """
    Keeps top Np n-plets for:
      - PR AUC with C as positive (pr_c_pos)
      - PR AUC with NR as positive (pr_nr_pos)

    We use two independent min-heaps:
      - heap_c: top Np by pr_c_pos
      - heap_nr: top Np by pr_nr_pos

    Returned lists are sorted descending by score.
    """

    def __init__(self, Np: int):
        self.Np = Np
        self.heap_c: List[Tuple[float, Tuple[int, ...]]] = []
        self.heap_nr: List[Tuple[float, Tuple[int, ...]]] = []

    def add(self, pr_c_pos: float, pr_nr_pos: float, nplet: Tuple[int, ...]):
        # --- Top Np for C-positive PR AUC ---
        if len(self.heap_c) < self.Np:
            heapq.heappush(self.heap_c, (pr_c_pos, nplet))
        elif pr_c_pos > self.heap_c[0][0]:
            heapq.heapreplace(self.heap_c, (pr_c_pos, nplet))

        # --- Top Np for NR-positive PR AUC ---
        if len(self.heap_nr) < self.Np:
            heapq.heappush(self.heap_nr, (pr_nr_pos, nplet))
        elif pr_nr_pos > self.heap_nr[0][0]:
            heapq.heapreplace(self.heap_nr, (pr_nr_pos, nplet))

    def get_results(self):
        """
        Returns:
            top_c:  list[(pr_c_pos, nplet)] sorted by pr_c_pos descending
            top_nr: list[(pr_nr_pos, nplet)] sorted by pr_nr_pos descending
        """
        top_c = sorted(self.heap_c, key=lambda x: -x[0])
        top_nr = sorted(self.heap_nr, key=lambda x: -x[0])
        return top_c, top_nr


def analyze_order(
    X_tensor: torch.Tensor,
    n_c: int,
    k: int,
    M: int,
    Np: int,
    batch_size: int = 2048,
    R: int = 82,
    device: torch.device = torch.device("cpu"),
    eval_fc=O_PR_AUC,
):
    """
    For a given order k:
      1) Sample M unique n-plets
      2) Evaluate PR AUC (C-positive and NR-positive) in batches
      3) Keep top Np by pr_c_pos and top Np by pr_nr_pos

    Returns:
        top_c, top_nr
        where each is a list of (score, nplet_tuple)
    """
    # 1) Sample unique n-plets
    print(f"Generating {M} unique nplets...")
    nplets_all = sample_unique_nplets(
        M=M,
        k=k,
        batch_size=100000,
        R=R,
        device=device,
    )  # shape: (M, k), torch.int16

    keeper = TailKeeperPR(Np)

    # 2) Loop over batches
    M_total = nplets_all.shape[0]
    # for start in range(0, M_total, batch_size):
    for start in tqdm(
        range(0, M_total, batch_size), desc=f"Evaluating n-plets (k={k})"
    ):
        end = min(start + batch_size, M_total)
        end = min(start + batch_size, M_total)
        nplets_batch = nplets_all[start:end]  # (b, k) torch.Tensor (int16)

        # Make sure it’s on the right device/type for compute_O_and_delta_batch
        nplets_batch_dev = nplets_batch.to(device=device, dtype=torch.long)

        pr_c_pos_batch, pr_nr_pos_batch = compute_O_and_delta_batch(
            X_tensor=X_tensor,
            nplets_batch=nplets_batch_dev,
            n_c=n_c,
            device=device,
            eval_fc=eval_fc,
        )
        # pr_*_batch are numpy arrays of length b

        # 3) Feed each n-plet into keeper
        for i in range(end - start):
            nplet_tuple = tuple(int(x) for x in nplets_batch[i].cpu().numpy().tolist())
            keeper.add(
                float(pr_c_pos_batch[i]),
                float(pr_nr_pos_batch[i]),
                nplet_tuple,
            )

    top_c, top_nr = keeper.get_results()
    return top_c, top_nr
