import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
import pickle
import chex
from typing import List, Dict

class GaussianMixtureModel:
    """
    A class to represent a Gaussian Mixture Model (GMM) that can be fitted to trajectory data and allows for density computation.
    
    Attributes
    ----------
    gms_means : List[jnp.ndarray]
        List to store means of the Gaussian components for each time step.
    gms_covs_invs : List[jnp.ndarray]
        List to store the inverses of the covariance matrices for each time step.
    gms_den : List[float]
        List to store normalization factors (density denominators) for each time step.
    gms_weights : List[float]
        List to store the weights of each Gaussian component for each time step.
    """
    
    def __init__(self):
        """
        Initializes the GaussianMixtureModel class.
        """
        self.gms_means: List[jnp.ndarray] = []
        self.gms_covs_invs: List[jnp.ndarray] = []
        self.gms_den: List[float] = []
        self.gms_weights: List[float] = []

    def fit(self, 
            trajectory: Dict[float, jnp.ndarray], 
            n_components: int) -> None:
        """
        Fits a Gaussian Mixture Model (GMM) to the given trajectory data.

        Parameters
        ----------
        trajectory : dict
            A dictionary where each key is a time step and each value is a 2D array (n_samples, n_features) of data points.
        n_components : int
            The number of clusters (components) to use in the GMM.
        """
        for _, val in trajectory.items():
            chex.assert_type(val, float)
            chex.assert_rank(val, 2)  # Check that each value in trajectory is a 2D array

        data_dim = list(trajectory.values())[0].shape[1]
        for label in sorted(trajectory.keys()):
            data = trajectory[label]
            gm = GaussianMixture(n_components=n_components, random_state=seed)
            gm.fit(data)

            # Discard components with small determinants
            covariances = gm.covariances_
            dets = jnp.asarray([jnp.linalg.det(covariances[i]) for i in range(n_components)])
            idxs = jnp.where(jnp.greater(dets, 1e-4))

            # Store density parameters
            self.gms_means.append(gm.means_[idxs])
            self.gms_covs_invs.append(jnp.linalg.inv(gm.covariances_[idxs]))
            self.gms_den.append(1 / jnp.sqrt((2 * jnp.pi) ** data_dim * dets[idxs]))
            self.gms_weights.append(jnp.asarray(gm.weights_[idxs] / jnp.sum(gm.weights_[idxs])))

    def to_file(self, 
                filename: str):
        """
        Saves the GMM model parameters to a file.

        Parameters
        ----------
        filename : str
            The file path to save the model to.
        """
        data = {
            'gms_means': self.gms_means,
            'gms_covs_invs': self.gms_covs_invs,
            'gms_den': self.gms_den,
            'gms_weights': self.gms_weights
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def from_file(self, 
                  filename: str):
        """
        Loads the GMM model parameters from a file.

        Parameters
        ----------
        filename : str
            The file path to load the model from.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.gms_means = data['gms_means']
            self.gms_covs_invs = data['gms_covs_invs']
            self.gms_den = data['gms_den']
            self.gms_weights = data['gms_weights']

    def gmm_density(self, t: int, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the GMM density for a given time step and data point.

        Parameters
        ----------
        t : int
            The time step to use for computing the GMM density.
        x : jnp.ndarray
            The data point (array of shape (n_features,)) for which to calculate the density.

        Returns
        -------
        jnp.ndarray
            The computed density value at the specified time and state.
        """
        diffs = x - self.gms_means[t]  # (n_components, dim)
        mahalanobis_terms = jnp.einsum('ij,ijk,ik->i', diffs, self.gms_covs_invs[t], diffs)  # (n_components,)
        exponent_terms = jnp.exp(-0.5 * mahalanobis_terms)  # (n_components,)
        weighted_terms = self.gms_weights[t] * self.gms_den[t] * exponent_terms  # (n_components,)
        result = jnp.sum(weighted_terms)  # Scalar value
        return jnp.clip(result, a_min=0.00001)