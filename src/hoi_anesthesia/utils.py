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
