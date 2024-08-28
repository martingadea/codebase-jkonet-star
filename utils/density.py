import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
import pickle

class GaussianMixtureModel:
    def __init__(self):
        """
        Initializes the Gaussian Mixture Model with empty lists for storing parameters.
        """
        self.gms_means = []
        self.gms_covs_invs = []
        self.gms_den = []
        self.gms_weights = []

    def fit(self, grouped_values, n_components: int):
        """
        Fits a Gaussian Mixture Model to the provided grouped values.

        Parameters
        ----------
        grouped_values : dict
            Dictionary where the trajectories are stored. The keys of the dictionary are the timesteps and the values are arrays of data points.
            Each array should have shape (n_samples, n_features).

        n_components : int
            Number of components (clusters) to use in the Gaussian Mixture Model.

        Notes
        -----
        This method fits a Gaussian Mixture Model to each group of data points separately.
        Components with small determinants are discarded to avoid numerical instability.
        """
        data_dim = list(grouped_values.values())[0].shape[1]
        for label in sorted(grouped_values.keys()):
            data = grouped_values[label]
            gm = GaussianMixture(n_components=n_components)
            gm.fit(data)

            # Discard components with small determinants
            covariances = gm.covariances_
            dets = jnp.asarray([jnp.linalg.det(covariances[i]) for i in range(n_components)])
            idxs = jnp.where(jnp.greater(dets, 1e-4))[0]

            # Store density parameters
            self.gms_means.append(gm.means_[idxs])
            self.gms_covs_invs.append(jnp.linalg.inv(gm.covariances_[idxs]))
            self.gms_den.append(1 / jnp.sqrt((2 * jnp.pi) ** data_dim * dets[idxs]))
            self.gms_weights.append(jnp.asarray(gm.weights_[idxs] / jnp.sum(gm.weights_[idxs])))

    def to_file(self, filename: str):
        """
        Saves the Gaussian Mixture Model to a file.

        Parameters
        ----------
        filename : str
            Path to the file where the model should be saved.
        """
        data = {
            'gms_means': self.gms_means,
            'gms_covs_invs': self.gms_covs_invs,
            'gms_den': self.gms_den,
            'gms_weights': self.gms_weights
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def from_file(self, filename: str):
        """
        Loads the Gaussian Mixture Model from a file.

        Parameters
        ----------
        filename : str
            Path to the file from which the model should be loaded.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.gms_means = data['gms_means']
            self.gms_covs_invs = data['gms_covs_invs']
            self.gms_den = data['gms_den']
            self.gms_weights = data['gms_weights']

    def gmm_density(self, t, x):
        """
        Computes the density of the Gaussian Mixture Model at a given time t and state x.

        Parameters
        ----------
        t : int
            Time index corresponding to the group of components to use for the density calculation.

        x : jnp.ndarray
            State (data point) at which the density should be computed. Should have shape (n_features,).

        Returns
        -------
        float
            Density of the Gaussian Mixture Model at the specified time and state.
        """
        diffs = x - self.gms_means[t]  # (n_components, dim)
        mahalanobis_terms = jnp.einsum('ij,ijk,ik->i', diffs, self.gms_covs_invs[t], diffs)  # (n_components,)
        exponent_terms = jnp.exp(-0.5 * mahalanobis_terms)  # (n_components,)
        weighted_terms = self.gms_weights[t] * self.gms_den[t] * exponent_terms  # (n_components,)
        result = jnp.sum(weighted_terms)  # Scalar value
        return jnp.clip(result, a_min=0.00001)