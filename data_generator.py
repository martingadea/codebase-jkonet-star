import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import potentials_all, interactions_all
from utils.sde_simulator import SDESimulator
from utils.density import GaussianMixtureModel
from utils.ot import compute_couplings
from utils.plotting import plot_couplings, plot_level_curves
from collections import defaultdict
from typing import Tuple, Union

def filename_from_args(args):
    """
    Generates a filename based on the arguments given.

    Parameters:
        args (argparse.Namespace): Arguments parsed from the command line.
        - See main() for the arguments.
    """

    # Generate filename
    filename = f"potential_{args.potential}_"
    filename += f"internal_{args.internal}_"
    filename += f"beta_{args.beta}_"
    filename += f"interaction_{args.interaction}_"
    filename += f"dt_{args.dt}_"
    filename += f"T_{args.n_timesteps}_"
    filename += f"dim_{args.dimension}_"
    filename += f"N_{args.n_particles}_"
    filename += f"gmm_{args.n_gmm_components}_"
    filename += f"seed_{args.seed}_"
    filename += f"split_{args.test_split}"
    
    return filename

def train_test_split(
    values: jnp.ndarray,
    sample_labels: jnp.ndarray,
    test_size: float = 0.4
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Splits the dataset into training and testing sets while preserving the distribution of labels.

    This function ensures that the proportion of each label in the dataset is preserved in both the
    training and testing subsets.

    Parameters
    ----------
    values : jnp.ndarray
        The data array to be split.
    sample_labels : jnp.ndarray
        The corresponding labels for the data. Contains the timestep
        linked to each value.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Defaults to 0.4.

    Returns
    -------
    tuple of jnp.ndarray
        A tuple containing:

        - Train values: Subset of the data for training.
        - Train labels: Corresponding labels for the training data.
        - Test values: Subset of the data for testing.
        - Test labels: Corresponding labels for the testing data.
    """
    unique_labels = np.unique(sample_labels)
    train_indices = []
    test_indices = []

    for label in unique_labels:
        indices = np.where(sample_labels == label)[0]
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    train_indices = jnp.array(train_indices)
    test_indices = jnp.array(test_indices)

    return values[train_indices], sample_labels[train_indices], values[test_indices], sample_labels[test_indices]

def generate_data_from_trajectory(folder: str, values: jnp.ndarray, sample_labels: jnp.ndarray,
                                  n_gmm_components: int = 10, batch_size: int = 1000, data_type: str = 'train') -> None:
    """
    Fits Gaussian Mixture Models (GMM) to the trajectory data, computes couplings,
    and saves the results to disk. This function also plots the data and saves the plots.

    Parameters
    ----------
    folder : str
        Directory where the data and plots will be saved.
    values : jnp.ndarray
        Array of trajectory data points.
    sample_labels : jnp.ndarray
        Array of sample labels corresponding to each data point.
    n_gmm_components : int, optional
        Number of components for the Gaussian Mixture Model (default is 10).
    data_type : str, optional
        Type of data being processed, either 'train' or 'test' (default is 'train').

    Returns
    -------
    None
    """
    sample_labels = [int(label) for label in sample_labels]
    # Group the values by sample labels
    trajectory = defaultdict(list)
    for value, label in zip(values, sample_labels):
        trajectory[label].append(value)

    # Convert lists to arrays
    trajectory = {label: jnp.array(values) for label, values in trajectory.items()}
    sorted_labels = sorted(trajectory.keys())

    # Check if the dataset is unbalanced (i.e., varying number of particles at each timestep)
    num_particles_per_step = [trajectory[label].shape[0] for label in sorted_labels]
    is_unbalanced = len(set(num_particles_per_step)) > 1

    if n_gmm_components > 0:
        print("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixtureModel()
        gmm.fit(trajectory, n_gmm_components)
        cmap = plt.get_cmap('Spectral')

        all_values = jnp.vstack([trajectory[label] for label in sorted_labels])
        x_min = jnp.min(all_values[:, 0]) * 0.9
        x_max = jnp.max(all_values[:, 0]) * 1.1
        y_min = jnp.min(all_values[:, 1]) * 0.9
        y_max = jnp.max(all_values[:, 1]) * 1.1

        for label in sorted_labels:
            # Plot particles
            plt.scatter(trajectory[label][:, 0], trajectory[label][:, 1],
                        c=[cmap(float(label) / len(sorted_labels))], marker='o', s=4)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(os.path.join('out', 'plots', folder, f'density_{label}.png'))
            plt.clf()

    print("Computing couplings...")

    for t, label in enumerate(sorted_labels[:-1]):
        next_label = sorted_labels[t + 1]
        values_t = trajectory[label]
        values_t1 = trajectory[next_label]

        # Compute couplings
        if is_unbalanced or batch_size == -1:
            couplings = compute_couplings(
                values_t,
                values_t1,
                next_label)
        else:
            couplings = []
            for i in range(int(jnp.ceil(trajectory[0].shape[0]/ batch_size))):
                idxs = jnp.arange(i * batch_size, min(
                    trajectory[0].shape[0], (i + 1) * batch_size
                ))
                couplings.append(compute_couplings(
                    trajectory[t][idxs, :],
                    trajectory[t + 1][idxs, :],
                    next_label
                ))
            couplings = jnp.concatenate(couplings, axis=0)
        jnp.save(os.path.join('data', folder, f'couplings_{data_type}_{label}_to_{next_label}.npy'), couplings)
        # Save densities and gradients
        ys = couplings[:, (couplings.shape[1] - 1) // 2:-2] #Changed the 2 to match the new shape of couplings
        rho = lambda x: 0.
        if n_gmm_components > 0:
            rho = lambda x: gmm.gmm_density(t, x)
        densities = jax.vmap(rho)(ys).reshape(-1, 1)
        densities_grads = jax.vmap(jax.grad(rho))(ys)
        data = jnp.concatenate([densities, densities_grads], axis=1)
        jax.numpy.save(os.path.join('data', folder, f'density_and_grads_{data_type}_{label}_to_{next_label}.npy'), data)
        
        # Plot couplings
        plot_couplings(couplings)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(os.path.join('out', 'plots', folder, f'couplings_{data_type}_{label}_to_{next_label}.png'))
        plt.clf()

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the data generation and processing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the following:

        - load_from_file (str): Path to a file to load a pre-generated trajectory.
          If provided, skips data generation.

        - potential (str): Name of the potential energy to use. Options include
          various potentials or 'none' to skip.

        - n_timesteps (int): Number of timesteps for the SDE simulation.

        - dt (float): Time increment for each step in the simulation.

        - internal (str): Type of internal energy ('wiener' for Wiener process
          or 'none').

        - beta (float): Standard deviation of the Wiener process. Used only if
          `internal` is 'wiener'.

        - interaction (str): Name of the interaction energy. Options include
          various interactions or 'none'.

        - dimension (int): Dimensionality of the system for synthetic data generation.

        - n_particles (int): Number of particles to simulate.

        - batch_size (int): Batch size for computing couplings.

        - n_gmm_components (int): Number of components in the Gaussian Mixture
          Model. Set to 0 to disable GMM fitting.

        - seed (int): Random seed for reproducibility.

        - test_split (float): Ratio for train-test split. Set to 0 for
          no split.

    Returns
    -------
    None
    """
    print("Running with arguments: ", args)
    key = jax.random.PRNGKey(args.seed)

    folder = filename_from_args(args) if args.load_from_file is None else args.load_from_file

    if not os.path.exists(os.path.join('data', folder)):
        os.makedirs(os.path.join('data', folder))
    if not os.path.exists(os.path.join('out', 'plots', folder)):
        os.makedirs(os.path.join('out', 'plots', folder))

    if args.load_from_file is None:
        sde_simulator = SDESimulator(
            args.dt,
            args.n_timesteps,
            1,
            potentials_all[args.potential] if args.potential != 'none' else False,
            args.beta if args.internal == 'wiener' else False,
            interactions_all[args.interaction] if args.interaction != 'none' else False
        )
        print("Generating data...")
        init_pp = jax.random.uniform(
            key, 
            (args.n_particles, args.dimension), minval=-4, maxval=4)
        trajectory = sde_simulator.forward_sampling(key, init_pp)

        data = trajectory.reshape(trajectory.shape[0] * trajectory.shape[1], trajectory.shape[2])
        sample_labels = jnp.repeat(jnp.arange(args.n_timesteps+1), trajectory.shape[1])
        # sample_labels = jnp.repeat(jnp.arange(args.n_timesteps + 1) * args.dt, trajectory.shape[1])
        jax.numpy.save(os.path.join('data', folder, 'data.npy'), data)
        jax.numpy.save(os.path.join('data', folder, "sample_labels.npy"), sample_labels)

        # Save args to file
        with open(os.path.join('data', folder, 'args.txt'), 'w') as file:
            file.write(f"potential={args.potential}\n")
            file.write(f"internal={args.internal}\n")
            file.write(f"beta={args.beta}\n")
            file.write(f"interaction={args.interaction}\n")
            file.write(f"dt={args.dt}\n")

        if args.potential != 'none':
            potential = potentials_all[args.potential]
            plot_level_curves(potential, ((-4, -4), (4, 4)),
                              save_to=os.path.join('out', 'plots', folder, 'level_curves_potential'))
        if args.interaction != 'none':
            interaction = interactions_all[args.interaction]
            plot_level_curves(interaction, ((-4, -4), (4, 4)),
                              save_to=os.path.join('out', 'plots', folder, 'level_curves_interaction'))
    else:
        print("Loading data from file...")
        folder = args.load_from_file
        data = jax.numpy.load(os.path.join('data', folder, 'data.npy'))
        sample_labels = jax.numpy.load(os.path.join('data', folder, 'sample_labels.npy'))

    # Perform train-test split
    if args.test_split != 0:
        train_values, train_labels, test_values, test_labels = train_test_split(data, sample_labels, test_size=args.test_split)
    else:
        train_values, train_labels = data, sample_labels

    # Generate data for train set
    jax.numpy.save(os.path.join('data', folder, 'train_data.npy'), train_values)
    jax.numpy.save(os.path.join('data', folder, 'train_sample_labels.npy'), train_labels)
    generate_data_from_trajectory(folder, train_values, train_labels, args.n_gmm_components, args.batch_size,
                                  data_type='train')

    if args.test_split != 0:
        # Generate data for test set
        jax.numpy.save(os.path.join('data', folder, 'test_data.npy'), test_values)
        jax.numpy.save(os.path.join('data', folder, 'test_sample_labels.npy'), test_labels)
        generate_data_from_trajectory(folder, test_values, test_labels, args.n_gmm_components, args.batch_size,
                                      data_type='test')

    print("Done.")


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load-from-file', 
        type=str, 
        default=None, 
        help="""
        Instead of generating a synthetic trajectory, load it from a file.

        The trajectory must be a numpy array of shape (n_timesteps + 1, n_particles, dimension).
        """
    )
    
    parser.add_argument(
        '--potential', 
        type=str, 
        default='none',
        choices=list(potentials_all.keys()) + ['none'],
        help="""Name of the potential energy to use.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--n-timesteps', 
        type=int, 
        default=5,
        help="""Number of timesteps of the simulation of the SDE.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--dt', 
        type=float, 
        default=0.01,
        help="""dt in the simulation of the SDE.
        
        Note: This parameter is considered only if --dataset is 'sde'.
        """
        )
    
    parser.add_argument(
        '--internal', 
        type=str, 
        default='none',
        choices=['wiener', 'none'],
        help="""Name of the internal energy to use.
        
        Note: 
            - This parameter is considered only if --dataset is 'sde'.
            - 'wiener' requires additionally the --sd parameter.
            - 'none' means no internal energy is considered.
            - At the moment only wiener process is implemented.
        """
        )
    
    parser.add_argument(
        '--beta', 
        type=float, 
        default=0.0,
        help="""Standard deviation of the wiener process. Must be positive.
        
        Note: This parameter is considered only if --internal is 'wiener'.
        """
        )
    
    parser.add_argument(
        '--interaction', 
        type=str, 
        default='none',
        choices=list(interactions_all.keys()) + ['none'],
        help="""Name of the interaction energy to use.
        
        Note: 
            - This parameter is considered only if --dataset is 'sde'.
            - 'none' means no internal energy is considered.
        """
        )
    
    parser.add_argument(
        '--dimension', 
        type=int, 
        default=2,
        help="""
        Dimensionality of the system. Used to generate synthetic data.
        """
        )
    
    parser.add_argument(
        '--n-particles', 
        type=int, 
        default=1000,
        help="""
        Number of particles sampled generated.
        """
        )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for computing the couplings.'
    )
    
    parser.add_argument(
        '--n-gmm-components',
        type=int,
        default=10,
        help='Number of components of the Gaussian Mixture Model. 0 for no GMM.'
    )
    
    # reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set seed for the run.'
    )

    #Train-test split
    parser.add_argument(
        '--test-split',
        type=float, 
        default=0,
        help='Train test split.'
        )

    args = parser.parse_args()

    main(args)
