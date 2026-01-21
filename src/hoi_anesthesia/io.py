from collections import defaultdict
import h5py


def load_covariance_dict(filepath, lazy=False):
    """
    Parameters
    ----------
    filepath : str
        Path to the .h5 file.
    lazy : bool
        If False (default)  → load every dataset into RAM (returns plain NumPy arrays).
        If True             → return h5py.Dataset handles (zero-copy); file must stay open.

    Returns
    -------
    covs : dict
        Two-level dict: covs[dataset][state] = ndarray | h5py.Dataset
    """
    covs = defaultdict(dict)

    if lazy:
        # keep file handle alive by attaching it to the dictionary itself
        h5f = h5py.File(filepath, "r")
        covs["_h5file"] = h5f  # so GC won't close it
        for dataset in h5f:
            for state in h5f[dataset]:
                covs[dataset][state] = h5f[dataset][state]  # h5py.Dataset
    else:
        with h5py.File(filepath, "r") as h5f:
            for dataset in h5f:
                for state in h5f[dataset]:
                    covs[dataset][state] = h5f[dataset][state][:]
    return covs


def print_time(t_i, t_f):
    elapsed_time_seconds = t_f - t_i
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    print("Elapsed time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))
