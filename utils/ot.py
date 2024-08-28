import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
import ot
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def wasserstein_couplings(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the optimal transport plan (couplings) between two sets of points xs and ys.

    Parameters
    ----------
    xs : jnp.ndarray
        An array of shape (n_samples_x, n_features) representing the first set of points.
    ys : jnp.ndarray
        An array of shape (n_samples_y, n_features) representing the second set of points.

    Returns
    -------
    jnp.ndarray
        The optimal transport plan matrix of shape (n_samples_x, n_samples_y).
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    M = ot.dist(xs, ys)

    return ot.emd(a, b, M, numItermax=1000000)

def wasserstein_loss(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """
    Computes transport between xs and ys.
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    M = ot.dist(xs, ys)

    return ot.emd2(a, b, M, numItermax=1000000)

@jax.jit
def sinkhorn_loss(xs: jnp.ndarray, ys: jnp.ndarray, epsilon: float = 1.0) -> float:
    """
    Computes the Sinkhorn divergence (a regularized Wasserstein distance) between two sets of points xs and ys.

    Parameters
    ----------
    xs : jnp.ndarray
        An array of shape (n_samples_x, n_features) representing the first set of points.
    ys : jnp.ndarray
        An array of shape (n_samples_y, n_features) representing the second set of points.
    epsilon : float, optional
        Regularization parameter for the Sinkhorn algorithm, by default 1.

    Returns
    -------
    float
        The Sinkhorn divergence between the two sets of points.
    """
    a = jnp.ones(xs.shape[0]) / xs.shape[0]
    b = jnp.ones(ys.shape[0]) / ys.shape[0]

    geom = pointcloud.PointCloud(xs, ys, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a, b)

    solver = sinkhorn.Sinkhorn()
    out = solver(prob)

    return out.reg_ot_cost


def compute_couplings(batch: jnp.ndarray, batch_next: jnp.ndarray, time: float):
    """
    Computes the couplings between particles in two consecutive batches.

    Parameters
    ----------
    batch : jnp.ndarray
        The array of particles at the current timestep with shape (n_particles, n_features).

    batch_next : jnp.ndarray
        The array of particles at the next timestep with shape (n_particles, n_features).

    time : float
        The time step of batch_next.

    Returns
    -------
    jnp.ndarray
        An array of shape (n_relevant_couplings, 2 * n_features + 2) where each row contains:
        - Particle from `batch` (shape: (n_features,))
        - Particle from `batch_next` (shape: (n_features,))
        - Time (float)
        - Coupling weight (float)

        Only the relevant couplings, where the weight is greater than a threshold, are included.
    """
    weights = wasserstein_couplings(batch, batch_next)

    # Create particle indices
    idx_t = jnp.arange(batch.shape[0])
    idx_t_next = jnp.arange(batch_next.shape[0])
    idx_t, idx_t_next \
        = jnp.meshgrid(idx_t, idx_t_next, indexing='ij')
    x = batch[idx_t.flatten()]
    y = batch_next[idx_t_next.flatten()]

    # Stack the columns so to have particle_x, particle_y, coupling_weight on each row
    couplings = jnp.column_stack((x, y, jnp.full_like(weights.flatten(), time), weights.flatten()))


    # Pick top couplings (~transport map)
    relevant_couplings = couplings[couplings[:, -1] > 1/ (10 * max(batch.shape[0],batch_next.shape[0]))]

    return relevant_couplings