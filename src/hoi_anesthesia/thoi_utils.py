from typing import Optional, List, Union, Callable

import logging
from thoi.heuristics.commons import _get_valid_candidates
from thoi.commons import _normalize_input_data
from typing import Optional, Callable, Union, List
from tqdm.auto import tqdm, trange
from functools import partial
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from thoi.typing import TensorLikeArray
from thoi.commons import _normalize_input_data
from thoi.measures.gaussian_copula import _get_tc_dtc_from_batched_covmat
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction
from thoi.commons import _get_string_metric
from functools import partial
import torch


def _get_bias_correctors(
    T: Optional[List[int]], order: int, batch_size: int, D: int, device: torch.device
):
    if T is not None:
        # |batch_size|
        bc1 = torch.tensor(
            [_gaussian_entropy_bias_correction(1, t) for t in T], device=device
        )
        bcN = torch.tensor(
            [_gaussian_entropy_bias_correction(order, t) for t in T], device=device
        )
        bcNmin1 = torch.tensor(
            [_gaussian_entropy_bias_correction(order - 1, t) for t in T], device=device
        )
    else:
        # |batch_size|
        bc1 = torch.tensor([0] * D, device=device)
        bcN = torch.tensor([0] * D, device=device)
        bcNmin1 = torch.tensor([0] * D, device=device)

    # |batch_size x D|
    bc1 = bc1.repeat(batch_size)
    bcN = bcN.repeat(batch_size)
    bcNmin1 = bcNmin1.repeat(batch_size)

    return bc1, bcN, bcNmin1


@torch.no_grad()
def random_sampler(
    N: int, order: int, repeat: int, device: Optional[torch.device] = None
):

    device = torch.device("cpu") if device is None else device

    return torch.stack(
        [torch.randperm(N, device=device)[:order] for _ in range(repeat)]
    )


def _generate_nplets_covmants_batched(
    covmats_batched: torch.Tensor, nplets_batched: torch.Tensor
) -> torch.Tensor:
    """
    covmats_batched: (bs, D, N, N)
    nplets_batched:  (bs, order)
    returns:         (bs, D, order, order)
    """
    bs, D, N, _ = covmats_batched.shape
    order = nplets_batched.shape[1]

    # (bs*D, N, N) + (bs*D, order)
    cov_bd = covmats_batched.reshape(bs * D, N, N)
    idx_bd = nplets_batched.repeat_interleave(D, dim=0)  # (bs*D, order)

    # select rows
    rows = torch.gather(
        cov_bd, 1, idx_bd.unsqueeze(-1).expand(-1, order, N)
    )  # (bs*D, order, N)

    # select cols
    cols_idx = idx_bd.unsqueeze(1).expand(-1, order, order)  # (bs*D, order, order)
    sub = torch.gather(rows, 2, cols_idx)  # (bs*D, order, order)

    return sub.view(bs, D, order, order)


@torch.no_grad()
def nplets_measures(
    covmats_pairs: torch.Tensor,  # [P, S, N, N]
    nplets_rpo: torch.Tensor,  # [R, P, order]
    *,
    T: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: int = logging.INFO,
    batch_size: int = 100000,
):
    """
    Aligned evaluation: for each repeat r and pair p, evaluate the single n-plet nplets[r,p,:]
    on exactly the S covariance matrices of pair p.

    Inputs
    ------
    covmats_pairs : [P, S, N, N]
    nplets        : [R, P, order]
    T             : int or length-S list (bias correction per dataset in a pair)

    Returns
    -------
    measures : [R, P, S, 4]  # (TC, DTC, O, S) per pair (S datasets)
    """

    logging.basicConfig(
        level=verbose, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # ---- validate & move ----
    covmats_pairs = torch.as_tensor(covmats_pairs, device=device).contiguous()
    nplets_rpo = torch.as_tensor(nplets_rpo, device=device).contiguous()

    assert covmats_pairs.ndim == 4, "covmats_pairs must have shape (P, D, N, N)"
    assert nplets_rpo.ndim == 3, "nplets_rpo must have shape (R, P, order)"

    P, D, N, _ = covmats_pairs.shape
    R, P2, order = nplets_rpo.shape
    assert P == P2, "Mismatch: covmats pairs P != nplets P"

    B = R * P
    nplets_batched = nplets_rpo.reshape(B, order)  # (B, order)
    # Repeat pairs across R and flatten to match (B, D, N, N)
    covmats_batched = (
        covmats_pairs.unsqueeze(0).expand(R, -1, -1, -1, -1).reshape(B, D, N, N)
    )

    # ---- precompute helpers for given order/S ----
    allmin1 = _all_min_1_ids(order, device=device)
    # We'll construct bias correctors per (batch of RP) × S
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device)

    dataloader = DataLoader(
        nplets_batched, batch_size=min(batch_size, B), shuffle=False
    )
    results = []
    offset = 0
    for nplet_batch in tqdm(
        dataloader, desc="Processing n-plets", leave=False, position=1
    ):
        bs = nplet_batch.shape[0]
        cov_batch = covmats_batched[offset : offset + bs]  # (bs, D, N, N)
        cov_batch.shape[:2]
        # Create the covariance matrices for each nplet in the batch
        # |curr_batch_size| x |D| x |order| x |order|
        nplets_covmats = _generate_nplets_covmants_batched(cov_batch, nplet_batch)
        nplets_covmats.shape
        # Pack covmats in a single batch
        # |curr_batch_size x D| x |order| x |order|
        nplets_covmats = nplets_covmats.view(bs * D, order, order)

        # Batch process all nplets at once
        m = _get_tc_dtc_from_batched_covmat(
            nplets_covmats,
            allmin1,
            bc1[offset * D : (offset + bs) * D],
            bcN[offset * D : (offset + bs) * D],
            bcNmin1[offset * D : (offset + bs) * D],
        )

        # Unpack results
        # |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|
        nplets_tc, nplets_dtc, nplets_o, nplets_s = m

        # Collect results
        results.append(
            torch.stack(
                [
                    nplets_tc.view(bs, D),
                    nplets_dtc.view(bs, D),
                    nplets_o.view(bs, D),
                    nplets_s.view(bs, D),
                ],
                dim=-1,
            )
        )

    # Concatenate all results
    return torch.cat(results, dim=0)


def _evaluate_nplets(
    covmats: torch.Tensor,
    T: Optional[List[int]],
    batched_nplets: torch.Tensor,
    metric: Union[str, Callable],
    batch_size: int,
    device: torch.device,
):
    """
    - covmats (torch.Tensor): The covariance matrix or matrixes with shape (N, N) or (D, N, N)
    - T (Optional[List[int]]): The number of samples for each multivariate series or None
    - batched_nplets (torch.Tensor): The nplets to calculate the inverse of the oinformation with shape (total_size, order)
    - metric (str): The metric to evaluate. One of tc, dtc, o, s or Callable
    - batch_size (int): The batch size to use for the computation
    - device (torch.device): The device to use
    """

    if len(covmats.shape) == 3:
        covmats = covmats.unsqueeze(0)
    metric_func = (
        partial(_get_string_metric, metric=metric)
        if isinstance(metric, str)
        else metric
    )
    # |bached_nplets| x |D| x |4 = (tc, dtc, o, s)|
    batched_measures = nplets_measures(
        covmats, nplets_rpo=batched_nplets, T=T, batch_size=batch_size, device=device
    )

    # |batch_size|
    return metric_func(batched_measures).to(device)


@torch.no_grad()
def simulated_annealing_parallel(
    X: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
    order: Optional[int] = None,
    *,
    covmat_precomputed: bool = False,
    T: Optional[Union[int, List[int]]] = None,
    initial_solution: Optional[torch.Tensor] = None,
    repeat: int = 10,
    batch_size: int = 1000000,
    device: torch.device = torch.device("cpu"),
    max_iterations: int = 1000,
    early_stop: int = 100,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.99,
    metric: Union[str, Callable] = "o",  # tc, dtc, o, s
    largest: bool = False,
    verbose: int = logging.INFO,
):

    logging.basicConfig(
        level=verbose, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    D = X.shape[0]  # number of datasets
    covmats = torch.empty_like(X)
    for d in range(D):
        covmats[d, :, :, :], S, N, T = _normalize_input_data(
            X[d, :, :, :],
            covmat_precomputed,
            T,
            device,
        )  # now S is number of subsets

    # Compute current solution
    # |batch_size| x |order|
    if initial_solution is None:
        current_solution = random_sampler(N, order, repeat * D, device)
    else:
        current_solution = initial_solution.to(device).contiguous()
    current_solution = current_solution.view(repeat, D, order)
    # |batch_size|
    # Loop over covmats in pairs

    current_energy = _evaluate_nplets(
        covmats,
        T,  # we evaluate all items
        current_solution,  # every subset must evaluate its corresponding nplet
        metric,
        batch_size=batch_size,
        device=device,
    )
    current_energy = current_energy.reshape(repeat, D)
    if not largest:
        current_energy = -current_energy

    # Initial valid candidates
    # |batch_size| x |N-order|
    valid_candidates = _get_valid_candidates(
        current_solution.view(-1, order), N, device
    )
    valid_candidates = valid_candidates.reshape(repeat, D, -1)

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution.clone()
    # |batch_size|
    best_energy = current_energy.clone()

    # Repeat tensor for indexing the current_solution
    # |repeat| x |1|
    i_repeat = torch.arange(repeat * D, device=device)

    no_progress_count = 0  # torch.zeros(D)
    pbar = trange(max_iterations, leave=True, position=0)
    metric_name = metric.__name__ if callable(metric) else metric
    if verbose != logging.WARNING:
        pbar = range(max_iterations)  # plain Python loop
    else:
        pbar = trange(max_iterations, leave=True, position=0)
    for _ in pbar:

        # Get function name if metric is a function
        if not isinstance(pbar, range):  # only if tqdm
            pbar.set_description(
                f"mean({metric_name}) = {(1 if largest else -1) * best_energy.mean()} - ES: {no_progress_count}"
            )

        # Generate new solution by modifying the current solution.
        # Generate the random indexes to change.
        # |batch_size| x |order|
        i_sol = torch.randint(0, current_solution.shape[2], (repeat, D), device=device)
        i_cand = torch.randint(0, valid_candidates.shape[2], (repeat, D), device=device)

        # Update current values by new candidates and keep the original
        # candidates to restore where the new solution is not accepted.
        current_candidates = torch.gather(
            current_solution, 2, i_sol.unsqueeze(-1)
        ).squeeze(-1)
        new_candidates = torch.gather(
            valid_candidates, 2, i_cand.unsqueeze(-1)
        ).squeeze(-1)
        _ = current_solution.scatter_(
            2, i_sol.unsqueeze(-1), new_candidates.unsqueeze(-1)
        )

        # Calculate energy of new solution
        # |batch_size|
        new_energy = _evaluate_nplets(
            covmats,
            T,  # we evaluate all items
            current_solution,  # every subset must evaluate its corresponding nplet
            metric,
            batch_size=batch_size,
            device=device,
        )
        new_energy = new_energy.reshape(repeat, D)
        if not largest:
            new_energy = -new_energy

        # Calculate change in energy
        # delca_energy > 0 means new_energy is bigger (more optimal) than current_energy
        # |batch_size|
        delta_energy = new_energy - current_energy

        # Determine if we should accept the new solution
        # |batch_size|
        temp_probas = torch.rand((repeat, D), device=device) < torch.exp(
            delta_energy / temp
        )

        improves = delta_energy > 0
        accept_new_solution = torch.logical_or(improves, temp_probas)
        # accept_new_solution = accept_new_solution.reshape(repeat, D)
        # Apply accepted updates directly

        current_solution = current_solution.view(repeat * D, order)
        current_candidates = current_candidates.view(repeat * D)
        valid_candidates = valid_candidates.view(repeat * D, -1)
        i_sol_flat = i_sol.flatten()
        i_cand_flat = i_cand.flatten()
        accept_new_solution_flat = accept_new_solution.flatten()
        current_solution[
            i_repeat[~accept_new_solution_flat], i_sol_flat[~accept_new_solution_flat]
        ] = current_candidates[~accept_new_solution_flat]

        # Update valid_candidate for the accepted answers as they are not longer valid candidates
        # |batch_size| x |N-order|
        valid_candidates[
            i_repeat[accept_new_solution_flat], i_cand_flat[accept_new_solution_flat]
        ] = current_candidates[accept_new_solution_flat]
        current_solution = current_solution.view(repeat, D, order)
        current_candidates = current_candidates.view(repeat, D)
        valid_candidates = valid_candidates.view(repeat, D, -1)

        current_energy[accept_new_solution] = new_energy[accept_new_solution]

        new_global_maxima = new_energy > best_energy

        # |batch_size| x |order|
        best_solution[new_global_maxima] = current_solution[new_global_maxima]

        # |batch_size|
        best_energy[new_global_maxima] = new_energy[new_global_maxima]

        # Cool down
        temp *= cooling_rate
        # Early stop
        if torch.any(new_global_maxima):
            no_progress_count = 0
        else:
            no_progress_count += 1

        if no_progress_count >= early_stop:
            logging.info("Early stop reached")
            break

    # If minimizing, then return score to its real value
    if not largest:
        best_energy = -best_energy

    return best_solution, best_energy
