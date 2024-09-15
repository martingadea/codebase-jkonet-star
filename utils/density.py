import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
import pickle
import chex

class GaussianMixtureModel:

    gms_means = []
    gms_covs_invs = []
    gms_den = []
    gms_weights = []

    def fit(self, trajectory: dict, n_components: int) -> None:
        """
        Fits a Gaussian Mixture Model to the provided trajectory.

        Parameters
        ----------
        trajectory : dict
            Dictionary where the trajectories are stored. The keys of the dictionary are the timesteps and the values are arrays of data points.
            Each array should have shape (n_samples, n_features).

        n_components : int
            Number of components (clusters) to use in the Gaussian Mixture Model.

        """
        for key, val in trajectory.items():
            chex.assert_type(val, float)
            chex.assert_rank(val, 2)  # Check that each value in trajectory is a 2D array

        data_dim = list(trajectory.values())[0].shape[1]
        for label in sorted(trajectory.keys()):
            data = trajectory[label]
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

    def gmm_density(self, t: int, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the density of the Gaussian Mixture Model at a given time t and state x.

        Parameters
        ----------
        t : int
            Time index corresponding to Gaussian Mixture Model at time-step t.

        x : jnp.ndarray
            State (data point) at which the density should be computed. Should have shape (n_features,).

        Returns
        -------
        jnp.ndarray
            Density of the Gaussian Mixture Model at the specified time and state.
        """
        diffs = x - self.gms_means[t]  # (n_components, dim)
        mahalanobis_terms = jnp.einsum('ij,ijk,ik->i', diffs, self.gms_covs_invs[t], diffs)  # (n_components,)
        exponent_terms = jnp.exp(-0.5 * mahalanobis_terms)  # (n_components,)
        weighted_terms = self.gms_weights[t] * self.gms_den[t] * exponent_terms  # (n_components,)
        result = jnp.sum(weighted_terms)  # Scalar value
        return jnp.clip(result, a_min=0.00001)