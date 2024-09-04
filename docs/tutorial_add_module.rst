Add model
====================

In this section we will walk through the main steps one has to follow in order to add a new model.

.. code-block:: python

        def load_dataset(self, dataset_name: str) -> CouplingsDataset:
            """
            Load and return a dataset based on the given dataset name.

            This method creates an instance of the `CouplingsDataset` class using the specified dataset name.

            Parameters
            ----------
            dataset_name : str
                The name of the dataset to load. This name is used to locate and initialize the dataset.

            Returns
            -------
            CouplingsDataset
                An instance of the `CouplingsDataset` class, which contains the loaded dataset.
            """
            return CouplingsDataset(dataset_name)

.. code-block:: python

        def create_state(self, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
            """
            Create initial training states for the potential, internal, and interaction models.

            Parameters
            ----------
            rng : jax.random.PRNGKey
                Random key for initialization.

            Returns
            -------
            Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
                Tuple containing the training states for the potential, internal, and interaction models.
            """
            # to allow for jit compilation
            # train states
            potential = create_train_state(
                rng, self.model_potential, get_optimizer(self.config_optimizer), self.data_dim)
            internal = create_train_state(
                rng, self.model_internal, get_optimizer(self.config_optimizer), 1)
            interaction = create_train_state(
                rng, self.model_interaction, get_optimizer(self.config_optimizer), self.data_dim)
            return potential, internal, interaction

.. code-block:: python
        def _train_step(
            self,
            state: Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState],
            xs: jnp.ndarray,
            ys: jnp.ndarray,
            t: jnp.ndarray,
            ws: jnp.ndarray,
            rho: jnp.ndarray,
            rho_grad: jnp.ndarray
        ) -> Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]:
            """
            Execute a training step by calculating gradients and updating model parameters.

            Parameters
            ----------
            state : Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]
                Training state containing potential, internal, and interaction models.
            xs : jnp.ndarray
                Initial particle distribution.
            ys : jnp.ndarray
                Target particle distribution.
            t : jnp.ndarray
                Time step of the target particle distribution.
            ws : jnp.ndarray
                Weights of the couplings.
            rho : jnp.ndarray
                Density values.
            rho_grad : jnp.ndarray
                Gradient of density values.

            Returns
            -------
            Tuple[jnp.ndarray, Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]]
                The loss value and the updated training states.
            """
            potential, internal, interaction = state
            loss, grads = jax.value_and_grad(
                    self.loss, argnums=(0, 1, 2))(
                        potential.params,
                        internal.params,
                        interaction.params,
                        xs, ys, ws, rho, rho_grad)
            potential = potential.apply_gradients(grads=grads[0])
            internal = internal.apply_gradients(grads=grads[1])
            interaction = interaction.apply_gradients(grads=grads[2])

            return loss, (potential, internal, interaction)

Add model to the list as well as to the get_model function at __init__ models.

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


