Add model
====================

In this section, we'll guide you through the key steps required to add a new model. As an example, we'll demonstrate
by adding a simple dummy model.

First of all we create a Dummy_model class. This class will be the one used to create the object "model" in the
training script.

.. code-block:: python

        class Dummy_model:
                def __init__(self, config: dict, data_dim: int, tau: float) -> None:
                self.data_dim = data_dim

                self.layers = config['energy']['model']['layers']
                self.config_optimizer = config['energy']['optim']

                # create energy models
                self.model_potential = MLP(self.layers)

Now we will go through the different methods the class must have. First of all, we need a method to load
the dataset. There are several dataset classes already built in depending on the format of the data
required by the solver. Refer to :mod:`dataset` module for more information on the different dataset classes.


For this tutorial we will use the CouplingsDataset which returns information of the couplings and the density
in addition to the particles.

.. code-block:: python

        def load_dataset(self, dataset_name: str) -> CouplingsDataset:
            return CouplingsDataset(dataset_name)

Next we define a method to create the state for our energy models. In this tutorial we will limit to the potential term.

.. code-block:: python

        def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
            potential = create_train_state(
                rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim)
            return potential, _, _

We generate a loss function. In this case, since it is a dummy model we just generated a loss function with no physical meaning.
When we minimize the loss and update the parameters the function will converge to a potential that always returns 0.

.. code-block:: python

        def loss(
            self,
            potential_params: dict,
            xs: jnp.ndarray,
            ys: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.abs(jnp.sum(self.model_potential.apply({'params': potential_params}, ys)))

We must define a train_step method. This function must contain the calculation of the loss and the gradients as well as
the following update of the state.

.. code-block:: python

        def train_step(
            self,
            state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
            sample: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:

            xs, ys, t, ws, rho, rho_grad = sample
            potential, internal, interaction = state
            loss, grads = jax.value_and_grad(
                    self.loss, argnums=(0, 1, 2))(
                        potential.params, xs, ys)
            potential = potential.apply_gradients(grads=grads[0])
            return loss, (potential, _, _)

Also very important, the model must be able to retrieve the different energy terms, regardless of whether
they are learnt or they are zero. We must include 3 methods to our class: get_potential, get_beta and
get_interaction.

.. code-block:: python

    def get_potential(self, state):
        potential, _, _ = state
        return lambda x: potential.apply_fn({'params': potential.params}, x)

    def get_beta(self, state):
        return 0.

    def get_interaction(self, state):
        return lambda x: 0.


Add model to the list as well as to the get_model function at __init__ models. This will be the string which must be
used to specify the solver.

.. code-block:: python

    class EnumMethod(Enum):
        JKO_NET_STAR = 'jkonet-star' # Solve with jkonet*, full generality.
        JKO_NET_STAR_POTENTIAL = 'jkonet-star-potential' # Fit only potential energy.
        JKO_NET_STAR_POTENTIAL_INTERNAL = 'jkonet-star-potential-internal' # Fit potential energy + wiener process.
        JKO_NET_STAR_TIME_POTENTIAL = 'jkonet-star-time-potential' #Fit only potential energy. Time varying potential.
        JKO_NET_STAR_LINEAR = 'jkonet-star-linear' # Solve with jkonet*, linear parametrization.
        JKO_NET_STAR_LINEAR_POTENTIAL = 'jkonet-star-linear-potential' # Solve with jkonet*, linear parametrization of only the potential and internal energies.
        JKO_NET_STAR_LINEAR_POTENTIAL_INTERNAL = 'jkonet-star-linear-potential-internal' # Solve with jkonet*, linear parametrization of potential and internal energies.
        JKO_NET_STAR_LINEAR_INTERACTION = 'jkonet-star-linear-interaction' # Solve with jkonet*, linear parametrization of interaction energy only.
        JKO_NET = 'jkonet' # Fit potential energy with JKOnet, see https://arxiv.org/abs/2106.06345.
        JKO_NET_VANILLA = 'jkonet-vanilla' # Fit potential energy with JKOnet, no ICNN
        JKO_NET_MONGE_GAP = 'jkonet-monge-gap' # Fit potential energy with JKOnet using Monge gap regularizer
        DUMMY = 'dummy-model'


.. code-block:: python

    def get_model(
        solver: EnumMethod,
        config: dict,
        data_dim: int,
        dt: float):


        if solver == EnumMethod.DUMMY:
            from models.jkonet_star import Dummy_model
            cls = Dummy_model

        # Other model retrieval logic here
        # ...

        return cls(config, data_dim, dt)

Finally, we must also add the colormap specific to the model to the style.yaml function. This will be used during the
plotting of the predictions.

.. code-block:: yaml

    # training
    groundtruth:
      light: '#F1F1F1'
      dark: '#C7B7A3'
      marker: 'o'

    jkonet-star:
      light: '#CDF5FD'
      dark: '#A0E9FF'
      marker: '+'

    dummy-model:
      light: '#FFC1C1'
      dark: '#FF6666'
      marker: '+'

As a summary, this is how the whole dummy-model class looks.

.. code-block:: python

    class Dummy_model:
        def __init__(self, config: dict, data_dim: int, tau: float) -> None:
            self.data_dim = data_dim

            self.layers = config['energy']['model']['layers']
            self.config_optimizer = config['energy']['optim']

            # create energy models
            self.model_potential = MLP(self.layers)

        def load_dataset(self, dataset_name: str) -> CouplingsDataset:
            return CouplingsDataset(dataset_name)

        def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
            potential = create_train_state(rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim)
            return potential, None, None

        def loss(
                self,
                potential_params: dict,
                xs: jnp.ndarray,
                ys: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.abs(jnp.sum(self.model_potential.apply({'params': potential_params}, ys)))

        def train_step(
                self,
                state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
                sample: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
            xs, ys, t, ws, rho, rho_grad = sample
            potential, _, _ = state
            loss, grads = jax.value_and_grad(
                self.loss, argnums=(0, 1, 2))(
                potential.params, xs, ys)
            potential = potential.apply_gradients(grads=grads[0])
            return loss, (potential, _, _)

        def get_potential(self, state):
            potential, _, _ = state
            return lambda x: potential.apply_fn({'params': potential.params}, x)

        def get_beta(self, state):
            return 0.

        def get_interaction(self, state):
            return lambda x: 0.
