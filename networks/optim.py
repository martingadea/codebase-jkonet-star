# source: https://github.com/bunnech/jkonet
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
from flax.core import freeze
import optax
from typing import Dict, Callable, Any



def get_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    """
    Returns an Optax optimizer object based on the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing optimizer configuration. Expected keys are:
        - 'optimizer': The name of the optimizer ('Adam' or 'SGD').
        - 'lr': Learning rate for the optimizer.
        - 'beta1': Beta1 parameter for the Adam optimizer.
        - 'beta2': Beta2 parameter for the Adam optimizer.
        - 'eps': Epsilon parameter for the Adam optimizer.
        - 'grad_clip': Optional maximum global norm for gradient clipping.

    Returns
    -------
    optax.GradientTransformation
        The configured Optax optimizer object.

    Raises
    ------
    NotImplementedError
        If the optimizer name is not supported.
    """
    optimizer_name = config['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optax.adam(learning_rate=config['lr'],
                               b1=config['beta1'], b2=config['beta2'],
                               eps=config['eps'])
    elif optimizer_name == 'SGD':
        optimizer = optax.sgd(learning_rate=config['lr'],
                              momentum=None, nesterov=False)
    else:
        raise NotImplementedError(
            f'Optimizer {optimizer_name} not supported yet!')

    if config['grad_clip']:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optimizer)
    return optimizer


def create_train_state(
        rng: jax.random.PRNGKey,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        input_shape: int
) -> train_state.TrainState:
    """
    Creates an initial `TrainState` for the given model and optimizer.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        Random key used for initializing the model parameters.
    model : nn.Module
        Flax model used for creating the initial state.
    optimizer : optax.GradientTransformation
        Optimizer object used for updating the model parameters.
    input_shape : int
        Shape of the input data used to initialize the model.

    Returns
    -------
    train_state.TrainState
        The initialized train state containing model parameters and optimizer.
    """
    params = model.init(rng, jnp.ones(input_shape))['params']
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

def create_train_state_from_params(
        model: nn.Module,
        params: Dict[str, Any],
        optimizer: optax.GradientTransformation
) -> train_state.TrainState:
    """
    Creates a `TrainState` from existing model parameters.

    Parameters
    ----------
    model : nn.Module
        Flax model used for creating the initial state.
    params : Dict[str, Any]
        Dictionary of model parameters.
    optimizer : optax.GradientTransformation
        Optimizer object used for updating the model parameters.

    Returns
    -------
    train_state.TrainState
        The train state containing the provided model parameters and optimizer.
    """
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def global_norm(updates: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the global norm of gradients across a nested structure of tensors.

    Parameters
    ----------
    updates : Dict[str, jnp.ndarray]
        Dictionary where values are tensors (e.g., gradients).

    Returns
    -------
    jnp.ndarray
        The global norm of the gradients.
    """
    return jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(updates)]))


def clip_weights_icnn(params):
    params = params.unfreeze()
    for k in params.keys():
        if (k.startswith('Wz')):
            params[k]['kernel'] = jnp.clip(params[k]['kernel'], a_min=0)

    return freeze(params)


def penalize_weights_icnn(params):
    penalty = 0
    for k in params.keys():
        if (k.startswith('Wz')):
            penalty += jnp.linalg.norm(jax.nn.relu(-params[k]['kernel']))
    return penalty
