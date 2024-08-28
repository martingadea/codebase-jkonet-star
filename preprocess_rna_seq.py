import os
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler



def main(args: argparse.Namespace) -> None:
    """
    Load, preprocess and save scRNA-seq PCA-transformed data for trajectory analysis.

    This function loads a dataset from the paper by Tong et al. (2020), whitens the data, selects the number of
    components to retain, optionally filters the data by timestep, and saves the processed data and corresponding labels.

    The dataset used in this script is sourced from:
    - Tong, A., Huang, J., Wolf, G., Van Dijk, D., & Krishnaswamy, S. (2020, November).
      TrajectoryNet: A dynamic optimal transport network for modeling cellular dynamics.
      In International Conference on Machine Learning (pp. 9526-9536). PMLR.


    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the following:

        - n_components (int): Number of PCA components to retain.

        - timestep_train (int): Timestep to filter the data (-1 to keep all).

    Returns
    -------
    None

    References
    ----------
    1. Tong, A., Huang, J., Wolf, G., Van Dijk, D., & Krishnaswamy, S. (2020, November).
       TrajectoryNet: A dynamic optimal transport network for modeling cellular dynamics.
       In International Conference on Machine Learning (pp. 9526-9536). PMLR.
    """
    data_file = os.path.join(".", "data", "TrajectoryNet", "eb_velocity_v5.npz")
    data_dict = np.load(data_file, allow_pickle=True)

    sample_labels = data_dict["sample_labels"]
    pca_embeddings = data_dict["pcs"]
    scaler = StandardScaler()
    scaler.fit(pca_embeddings)
    pca_embeddings = scaler.transform(pca_embeddings)
    n_components = args.n_components
    data = pca_embeddings[:, :n_components]

    folder = f"data/RNA_PCA_{args.n_components}"
    time_step_train = args.timestep_train

    if time_step_train !=- 1:
        folder += f"_pred_{time_step_train}"
        mask = (sample_labels == time_step_train) | (sample_labels == time_step_train-1)
        data = data[mask]
        sample_labels = sample_labels[mask]

    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "data.npy"), data)
    np.save(os.path.join(folder, "sample_labels.npy"), sample_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n-components',
        type=int,
        default=5,
        help=f"""Number of components to keep in PCA.""",
        )
    parser.add_argument(
        '--timestep_train',
        type=int,
        choices=[-1, 1, 2, 3, 4,],
        default=-1,
        help=f"""Option to reduce the dataset to just one timestep.""",
    )
    args = parser.parse_args()
    main(args)