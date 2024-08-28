import glob
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from utils.functions import potentials_all, interactions_all
from utils.ot import wasserstein_loss
from utils.sde_simulator import get_SDE_predictions
from utils.plotting import plot_predictions

from collections import defaultdict
from typing import Tuple, Optional, Callable, List


class PopulationDataset(Dataset):
    """
    Dataset class for loading and accessing particle trajectory data.

    The dataset is expected to be located in a directory named 'data/{dataset_name}'
    and consist of a single .npy file named 'data.npy'. The data contains particle
    trajectories over time, where each timestep has a set of particles.

    Attributes
    ----------
    trajectory : np.ndarray
        Array of shape (num_timesteps, num_particles, num_features) containing
        the particle trajectories. Each entry in the array represents a particle's
        state at a given timestep.
    """
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the PopulationDataset by loading data from 'data.npy'.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load. The dataset should be located in
            'data/{dataset_name}' and should contain a .npy file named 'data.npy'.
        """
        self.trajectory = np.load(os.path.join('data', dataset_name, 'data.npy'))

    def __len__(self) -> int:
        """
        Returns the number of timesteps in the dataset.

        Returns
        -------
        int
            The number of timesteps in the dataset.
        """
        return self.trajectory.shape[1]

    def __getitem__(self, idx: int) -> list:
        """
        Retrieve particle data for each timestep at the given index.

        Parameters
        ----------
        idx : int
            The index of the particle to retrieve.

        Returns
        -------
        list of np.ndarray
            A list where each element is an array representing the state of a
            particle at each timestep. The length of the list corresponds to the
            number of timesteps, and each array represents the particle state
            at a specific timestep.
        """
        # returns a particle for each timestep
        # batching means getting more particles per timestep
        return [self.trajectory[t, idx, :] 
                for t in range(self.trajectory.shape[0])]


class CouplingsDataset(Dataset):
    """
    Dataset class for loading and accessing couplings data.

    The dataset is expected to be located in a directory named 'data/{dataset_name}'
    and consist of multiple .npy files. It provides access to input features, target features,
    time labels, weights, density values, and density gradients.

    Attributes
    ----------
    weight : np.ndarray
        Array of weights extracted from the couplings data.
    x : np.ndarray
        Array of input features extracted from the couplings data.
    y : np.ndarray
        Array of target features extracted from the couplings data.
    time : np.ndarray
        Array of time labels extracted from the couplings data.
    densities : np.ndarray
        Array of density values extracted from the densities files.
    densities_grads : np.ndarray
        Array of gradients of densities extracted from the densities files.
    """
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the CouplingsDataset by loading data from .npy files.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load. The dataset is expected to be located in a
            directory named 'data/{dataset_name}' and consist of multiple .npy files.
        """
        # load couplings for all timesteps together
        couplings = np.concatenate([np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_train_*.npy'))])
        self.weight = couplings[:, -1]
        self.x = couplings[:, :(couplings.shape[1] - 2) // 2]
        self.y = couplings[:, (couplings.shape[1] - 2) // 2:-2]
        self.time = couplings[:, -2]
        self.densities = np.concatenate(
            [np.load(f) for f in glob.glob(
                os.path.join('data', dataset_name, 'density_and_grads_train_*.npy'))]
        )
        self.densities_grads = self.densities[:, 1:]
        self.densities = self.densities[:, 0]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        """
        Retrieve a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - Input features (jnp.ndarray): Initial particle distribution.
            - Target features (jnp.ndarray): Target particle distribution.
            - Time label (jnp.ndarray): Time label.
            - Weight of the coupling (jnp.ndarray): Weight of the coupling.
            - Density value (jnp.ndarray): Density value.
            - Gradient of densities (jnp.ndarray): Gradient of densities.
        """
        return self.x[idx], self.y[idx], self.time[idx], self.weight[idx], self.densities[idx], self.densities_grads[
            idx]
    
class LinearParametrizationDataset(Dataset):
    """
    This dataset class loads and organizes data necessary for linear parametrization
    solver tasks. The data is expected to be located in the 'data/{dataset_name}'
    directory and consists of multiple .npy files for couplings and densities.

    Attributes
        ----------
        data : list of tuple of np.ndarray containing information about the couplings
            A list where each element is a tuple containing:
            - Input features (np.ndarray)
            - Target features (np.ndarray)
            - Time label (np.ndarray)
            - Weight of the coupling (np.ndarray)
            - Density values (np.ndarray)
            - Gradient of densities (np.ndarray)

    """
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the LinearParametrizationDataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.

        """
        couplings = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'couplings_train_*.npy'))]

        densities = [np.load(f) for f in glob.glob(
            os.path.join('data', dataset_name, 'density_and_grads_train_*.npy'))]
        self.data = [(
            c[:, :(c.shape[1] - 1) // 2], 
            c[:, (c.shape[1] - 1) // 2:-2],
            c[:, -2],
            c[:, -1],
            densities[t][:,0],
            densities[t][:,1:]
        ) for t, c in enumerate(couplings)]
        
    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        For this dataset, the number of elements is always 1 because the dataset
        is treated as a single entity.

        Returns
        -------
        int
            The number of elements (always 1 for this dataset).
        """
        return 1
    
    def __getitem__(self, _)-> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieve the entire dataset.

        Since this dataset is loaded as a single entity, this method returns all
        data at once. The parameter `_` is ignored.

        Parameters
        ----------
        _ : any
            This parameter is ignored.

        Returns
        -------
        list of tuple of np.ndarray
            The entire dataset as a list of tuples. Each tuple contains:
            - np.ndarray : Input features.
            - np.ndarray : Target features.
            - np.ndarray : Time label.
            - np.ndarray : Weight associated to the coupling.
            - np.ndarray : Density values.
            - np.ndarray : Gradient of densities.
        """
        return self.data
    
class PopulationEvalDataset(Dataset):
    potential: str = 'none'
    internal: str = 'none'
    beta: float = 0.0
    interaction: str = 'none'
    dt: float = 1.0
    T: int = 0
    data_dim: int = 0

    """
    This dataset class loads and organizes population trajectory data for evaluation.
    The dataset supports evaluation on test or training data, depending on the label.
    
    Attributes
        ----------
        trajectory : dict
            A dictionary where each key corresponds to a unique timestep in the dataset, and
            the value is an array of trajectory data associated with that timestep.
        label_mapping : dict
            A dictionary mapping the original sample labels to consecutive integer indices.
        T : int
            The number of timesteps in the trajectories.
        data_dim : int
            The dimensionality of the data at each timestep.
        no_ground_truth : bool
            Flag indicating if the dataset lacks a ground truth file.
        potential : str
            The potential function used in the predictions.
        internal : str
            The internal dynamics setting used.
        beta : float
            The beta parameter used in the simulations.
        interaction : str
            The interaction function used in the predictions.
        dt : float
            The timestep size used in the simulation.
        trajectory_only_potential : np.ndarray
            Trajectory predictions considering only the potential term.
        trajectory_only_interaction : np.ndarray
            Trajectory predictions considering only the interaction term.
    """
    def __init__(self, key, dataset_name: str, solver: str, label='test_data'):
        """
        Initialize the PopulationEvalDataset.

        This dataset class loads and organizes population trajectory data for evaluation.
        The dataset supports evaluation on test or training data, depending on the label.

        Parameters
        ----------
        key : Any
            A key used for random number generation or seeding.
        dataset_name : str
            The name of the dataset to load. The data should be located in the directory
            'data/{dataset_name}' and consist of .npy files.
        solver : str
            The solver method used, primarily for plotting or prediction purposes.
        label : str, optional
            Specifies whether to load 'test_data' or 'train_data'. Default is 'test_data'.

        """
        self.key = key
        self.solver = solver
        if label == 'test_data':
            data = np.load(os.path.join('data', dataset_name, 'test_data.npy'))
            sample_labels = np.load(os.path.join('data', dataset_name, 'test_sample_labels.npy'))
        else:
            data = np.load(os.path.join('data', dataset_name, 'train_data.npy'))
            sample_labels = np.load(os.path.join('data', dataset_name, 'train_sample_labels.npy'))

        unique_labels = np.unique(sample_labels)
        self.label_mapping = {original: i for i, original in enumerate(unique_labels)}

        self.trajectory = defaultdict(list)
        for value, label in zip(data, sample_labels):
            self.trajectory[self.label_mapping[label]].append(value)
        for label in self.trajectory:
            self.trajectory[label] = np.array(self.trajectory[label])
            self.data_dim = self.trajectory[label].shape[1]
        self.T = len(self.trajectory.keys())-1
        self.no_ground_truth = False
        try:
            with open(os.path.join('data', dataset_name, 'args.txt'), 'r') as file:
                for line in file:
                    if "potential" in line:
                        self.potential = line.split("=")[1][:-1]
                    elif "internal" in line:
                        self.internal = line.split("=")[1][:-1]
                    elif "beta" in line:
                        self.beta = float(line.split("=")[1][:-1])
                    elif "interaction" in line:
                        self.interaction = line.split("=")[1][:-1]
                    elif "dt" in line:
                        self.dt = float(line.split("=")[1][:-1])
            self.trajectory_only_potential = self._compute_separate_predictions(
                potentials_all[self.potential] if self.potential != 'none' else False,
                False,
                False)
            self.trajectory_only_interaction = self._compute_separate_predictions(
                False,
                False,
                interactions_all[self.interaction] if self.interaction != 'none' else False)
        except FileNotFoundError:
            print(f"Dataset {dataset_name} does not have a ground truth file. Skipping error computation.")
            self.no_ground_truth = True

    def _compute_separate_predictions(
            self,
            potential: Callable[[jnp.ndarray], float],
            beta: float,
            interaction: Callable[[jnp.ndarray], float]
    ) -> jnp.ndarray:
        """
        Compute separate predictions based on potential, beta, and interaction.

        This method computes the trajectory predictions for the population based on the
        specified potential function, beta parameter, and interaction function.

        Parameters
        ----------
        potential : Callable[[jnp.ndarray], float]
            A function representing the potential term, which takes a jnp.ndarray as input
            and returns a float value.
        beta : float
            The beta parameter used in the predictions.
        interaction : Callable[[jnp.ndarray], float]
            A function representing the interaction term, which takes a jnp.ndarray as
            input and returns a float value.

        Returns
        -------
        jnp.ndarray
            The predicted trajectories for the population based on the specified
            potential, beta, and interaction.
        """
        return get_SDE_predictions(
                    self.solver,
                    self.dt,
                    self.T,
                    1,
                    potential,
                    beta,
                    interaction,
                    self.key,
                    self.trajectory[0])


    def __len__(self) -> int:
        """
        Get the number of particles at the first timestep.

        Returns
        -------
        int
            The number of particles at the first timestep.
        """
        return self.trajectory[0].shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieves a particle's features at the first timestep.

        Parameters
        ----------
        idx : int
            The index of the particle to retrieve.

        Returns
        -------
        np.ndarray
            The features of the specified particle at the first timestep.
        """
        return self.trajectory[0][idx, :]

    def error_wasserstein(self, trajectory_predicted: np.ndarray) -> float:
        """
        Compute the Wasserstein loss between the predicted and true trajectories.

        This method calculates the Wasserstein distance (a measure of distance
        between probability distributions) between the predicted trajectories
        and the true trajectories over all timesteps.

        Parameters
        ----------
        trajectory_predicted : np.ndarray
            The predicted trajectory with shape (T, n_particles, n_features).

        Returns
        -------
        float
            The cumulative Wasserstein error over all timesteps.
        """
        error = 0
        for t in range(1, trajectory_predicted.shape[0]):
            error += wasserstein_loss(
                        trajectory_predicted[t], jnp.asarray(self.trajectory[t]))
        return error
    
    def error_potential(self, trajectory_predicted: np.ndarray) -> float:
        """
        Compute the mean squared error between the predicted trajectory and
        the ground truth trajectory predicted using only the potential function.

        This method calculates the error by comparing the predicted trajectory
        to a reference trajectory generated using only the potential function.

        Parameters
        ----------
        trajectory_predicted : np.ndarray
            The predicted trajectory with shape (T, n_particles, n_features).

        Returns
        -------
        float
            The mean squared error considering only the potential.
        """
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_potential) ** 2, axis=(0, 2)))
    
    def error_internal(self, beta_predicted: float) -> float:
        """
        Compute the error in the internal parameter (beta).

        This method calculates the difference between the predicted beta value
        and the true beta value, scaled by the trajectory length and time step.

        Parameters
        ----------
        beta_predicted : float
            The predicted beta value.

        Returns
        -------
        float
            The error in the internal parameter, scaled by the trajectory length and time step.
        """
        return np.sqrt(2) * np.abs(np.abs(beta_predicted) - np.abs(self.beta)) * self.T * self.dt
    
    def error_interaction(self, trajectory_predicted: np.ndarray) -> float:
        """
        Compute the mean squared error between the predicted trajectory and
        the ground truth trajectory predicted using only the interaction function.

        This method calculates the error by comparing the predicted trajectory
        to a reference trajectory generated using only the interaction function.

        Parameters
        ----------
        trajectory_predicted : np.ndarray
            The predicted trajectory with shape (T, n_particles, n_features).

        Returns
        -------
        float
            The mean squared error considering only the interaction.
        """
        return np.mean(np.sum((trajectory_predicted - self.trajectory_only_interaction) ** 2, axis=(0, 2)))

    def error_wasserstein_one_step_ahead(
            self,
            potential: Callable[[jnp.ndarray], float],
            beta: float,
            interaction: Callable[[jnp.ndarray], float],
            key_eval: jnp.ndarray,
            model: str,
            plot_folder_name: str
    ) -> jnp.ndarray:
        """
        Compute the Wasserstein error for one-step-ahead predictions.

        This method evaluates the prediction error by computing the Wasserstein distance
        between the predicted trajectory and the actual trajectory at each time step.

        Parameters
        ----------
        potential : Callable[[jnp.ndarray], float]
            Function that computes the potential based on a JAX array input.
        beta : float
            Beta parameter used in the predictions.
        interaction : Callable[[jnp.ndarray], float]
            Function that computes the interaction based on a JAX array input.
        key_eval : jnp.ndarray
            Random key for JAX-based random number generation.
        model : str
            Name of the solver model used. This is primarily used for plotting purposes.
        plot_folder_name : str
            Directory path where plots should be saved. If set to an empty string or None,
            no plots will be saved.

        Returns
        -------
        jnp.ndarray
            An array of Wasserstein errors for the one-step-ahead predictions over time steps.
            The array has length `T`, where each entry corresponds to the error at a specific
            time step.
        """
        error_wasserstein_one_ahead = jnp.ones(self.T)
        for t in range(self.T):
            init = self.trajectory[t]
            predictions = get_SDE_predictions(
                    self.solver,
                    self.dt,
                    1,
                    t+1,
                    potential,
                    beta,
                    interaction,
                    key_eval,
                    init)
            if plot_folder_name:
                plot_filename = f'one_ahead_tp_{t + 1}'
                plot_path = os.path.join(plot_folder_name, plot_filename)
                prediction_fig = plot_predictions(
                    predictions[-1].reshape(1, -1, self.data_dim),
                    self.trajectory,
                    interval=(t + 1, t + 1),
                    model=model,
                    save_to=plot_path)
                plt.close(prediction_fig)
            error_wasserstein_one_ahead = error_wasserstein_one_ahead.at[t].set(
                wasserstein_loss(predictions[-1], jnp.asarray(self.trajectory[t + 1])))
        return error_wasserstein_one_ahead

    def error_wasserstein_cumulative(
        self,
        predictions: jnp.ndarray,
        model: str,
        plot_folder_name: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Compute the cumulative Wasserstein error per timestep.

        This method calculates the Wasserstein distance between the predicted and actual
        trajectories at each timestep and returns the cumulative error.

        Parameters
        ----------
        predictions : jnp.ndarray
            Array of predicted trajectories with shape (T+1, n_particles, n_features).
            The predictions should cover the entire timespan from 0 to T.

        model : str
            Name of the solver model used. This is primarily used for plotting purposes.

        plot_folder_name : Optional[str], default=None
            Directory path where plots should be saved. If None, no plots will be saved.

        Returns
        -------
        jnp.ndarray
            Array of cumulative Wasserstein errors, with each entry corresponding to
            the error at a specific timestep. The array has length `T`.
        """
        error_wasserstein_cumulative = jnp.ones(self.T)
        for t in range(1, self.T + 1):
            if plot_folder_name:
                plot_path = os.path.join(plot_folder_name, f'cum_tp_{t}')
                trajectory_fig = plot_predictions(
                    predictions[t].reshape(1, -1, self.data_dim),
                    self.trajectory,
                    interval=(t, t),
                    model=model,
                    save_to=plot_path)
                plt.close(trajectory_fig)
            error_wasserstein_cumulative = error_wasserstein_cumulative.at[t - 1].set(
                wasserstein_loss(predictions[t], jnp.asarray(self.trajectory[t])))
        return error_wasserstein_cumulative