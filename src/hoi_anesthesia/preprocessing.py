import h5py
from thoi.commons import _normalize_input_data


def generate_covmats(data_dict, results_path, covmat_name="covariance_matrices_gc"):
    # Create HDF5 file
    with h5py.File(f"{results_path}/{covmat_name}.h5", "w") as h5f:
        for dataset_name, state_dict in data_dict.items():
            dset_grp = h5f.create_group(dataset_name)
            for state_name, state_data in state_dict.items():
                # _normalize_input_data handles a *batch*:
                # returns covmats : (N, R, R); R regions, N subjects
                covmats, D, Nvars, T = _normalize_input_data(state_data)
                # One HDF5 dataset per STATE
                _ = dset_grp.create_dataset(
                    state_name,  # path: /dataset/state
                    data=covmats,  # (N, 82, 82)
                    compression="gzip",
                    chunks=True,
                )
                # state_data  -  shape (N, 500, 82): N subjects, 500 samples, 82 regions
                # covmat  -  shape (N, 82, 82): N subjects, 82x82 covariances
                print(
                    f"{dataset_name}/{state_name} - {state_data.shape}  →  {covmats.shape}"
                )
    print(f"covmat dictionary saved at {results_path}/{covmat_name}.h5")
