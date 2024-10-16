Using JKOnet\*
==============

This guide provides an overview of how to use JKOnet\* to train models on synthetic data. The guide is divided into the following sections:

1. Generating synthetic data
2. Training JKOnet\* (and other models)

For reproducing the experiments in the paper, see the :doc:`benchmarks` page, and for more examples (including training JKOnet\* on the single-cell RNA dataset), see the :doc:`tutorials` page.

Generating synthetic data 🧩
-----------------------------

You can generate synthetic data using the `data_generator.py` script. The script generates data for a given potential energy, interaction energy, and internal energy. The script also computes the couplings and fits the densities, which are required for training JKOnet\*.

Example 1:
~~~~~~~~~~~

To generate population data driven by a potential energy function (e.g., the ``wavy_plateau`` function), run the following command:

.. code-block:: bash

   python data_generator.py --potential wavy_plateau

Example 2:
~~~~~~~~~~~

If you wish to include also internal and interaction energies, you can specify them as follows:

.. code-block:: bash

   python data_generator.py --potential wavy_plateau --interaction flowers --internal wiener --beta 0.1

Other parameters
~~~~~~~~~~~~~~~~
The `data_generator.py` script accepts the following parameters for customizing the data generation and performing ablations on various datasets:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Description
   * - ``--load-from-file``
     - Load pre-generated trajectory data from a file. Expects a NumPy array of shape ``(n_timesteps + 1, n_particles, dimension)``. If not provided, the script generates synthetic data. See :doc:`tutorial_rna` for an example.
   * - ``--potential``
     - Specifies the potential energy to be used for the SDE simulation. Examples include ``wavy_plateau``, ``styblinski_tang``, or ``none`` if no potential is desired. See :mod:`utils.functions` for more options.
   * - ``--n-timesteps``
     - Number of timesteps in the SDE simulation. Defines the length of the particle trajectory.
   * - ``--dt``
     - Time step size in the SDE simulation. Controls how often particles are updated in time.
   * - ``--internal``
     - Specifies the type of internal energy, such as ``'wiener'``, to simulate internal particle dynamics. Use ``'none'`` to disable internal energy.
   * - ``--beta``
     - Standard deviation of the Wiener process (used with ``--internal wiener``), defining the strength of the internal energy.
   * - ``--interaction``
     - Specifies interaction energy between particles, such as ``'flowers'``. Use ``'none'`` if no interaction energy is needed. See :mod:`utils.functions` for more options.
   * - ``--dimension``
     - Dimensionality of the system (e.g., 2D, 3D). Determines how many spatial dimensions the particles can move in.
   * - ``--n-particles``
     - Number of particles to simulate in the dataset. More particles increase the complexity and size of the data.
   * - ``--batch-size``
     - Batch size for computing couplings during the processing phase. Negative values disable batching.
   * - ``--n-gmm-components``
     - Number of components in the Gaussian Mixture Model (GMM) fitted to the data. Setting to 0 disables GMM fitting.
   * - ``--seed``
     - Seed for random number generation to ensure reproducibility of results.
   * - ``--test-ratio``
     - Proportion of the data to be allocated to the test set during the train-test split. Values range from 0 to 1.
   * - ``--split-trajectories``
     - If set, the train-test split is performed on entire trajectories, preserving continuity between time steps.
   * - ``--leave-one-out``
     - Leaves one time point out from the training data when set to a non-negative integer.

For more information on the ``data_generator.py`` script, see the :mod:`data_generator` module.

The script saves the generated data in the ``data/`` directory by default. The directory name containing the generated data includes the potential, internal, interaction, and the other parameters. In the following, let ``dataset_name`` be the name of the directory containing the generated data.


Training JKOnet\* 🚀
-----------------------------

To train JKOnet\* on the generated data, use the ``train.py`` script. The script trains a model using the JKOnet\* architecture and evaluates it on the test set.

For more information on the ``train.py`` script, see the :mod:`train` module.
For more information on the available models, see the :doc:`models` page and check the `paper <https://arxiv.org/abs/2406.12616>`__.

Example 1:
~~~~~~~~~~~

To train the JKOnet\* modeling only the potential energy on the generated data, run the following command:

.. code-block:: bash

   python train.py --solver jkonet-star-potential --dataset dataset_name


Available solvers
~~~~~~~~~~~~~~~~~

The following solvers (models) are available for training with JKOnet\*. Each solver corresponds to a different model configuration or variation:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Solver
     - Description
   * - ``jkonet-star``
     - JKOnet* with full generality, modeling all energy components (potential, internal, and interaction).
   * - ``jkonet-star-potential``
     - Fits only the potential energy component.
   * - ``jkonet-star-potential-internal``
     - Fits both the potential energy and Wiener process (internal energy).
   * - ``jkonet-star-time-potential``
     - Fits the potential energy with time-dependent features.
   * - ``jkonet-star-linear``
     - JKOnet* using a linear parametrization for potential, internal, and interaction energies.
   * - ``jkonet-star-linear-potential``
     - JKOnet* using linear parametrization for potential energy only (no interaction or internal energy).
   * - ``jkonet-star-linear-potential-internal``
     - JKOnet* using linear parametrization for both potential and internal energies.
   * - ``jkonet-star-linear-interaction``
     - JKOnet* using linear parametrization for interaction energy only.
   * - ``jkonet``
     - Standard JKOnet model for fitting potential energy, as described in the `paper <https://arxiv.org/abs/2106.06345>`_.
   * - ``jkonet-vanilla``
     - JKOnet model without using Input Convex Neural Networks (ICNN).
   * - ``jkonet-monge-gap``
     - JKOnet with Monge gap regularization.

To add a custom solver, see the :doc:`tutorial_add_module` page.

Other parameters
~~~~~~~~~~~~~~~~

The ``train.py`` script accepts the following parameters for customizing the training process:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Description
   * - ``--solver``
     - Name of the solver (model) to use.
   * - ``--dataset``
     - Name of the dataset to train the model on. The dataset should be prepared and located in a directory matching this name.
   * - ``--eval``
     - Option to test the fit on ``train_data`` or ``test_data`` (e.g., for debugging purposes). Default is ``test_data``.
   * - ``--wandb``
     - If specified, activates Weights & Biases logging for experiment tracking.
   * - ``--debug``
     - If specified, runs the script in debug mode (disables JIT compilation in JAX for easier debugging).
   * - ``--seed``
     - Seed for random number generation to ensure reproducibility.

Config file
~~~~~~~~~~~~

This configuration file contains the settings for training and evaluating the energy model. The file is divided into several sections:

1. **Training settings**: Specifies evaluation frequency, batch size, total epochs, and whether to save outputs locally.

2. **Metrics configuration**: Specifies the evaluation metrics.

3. **Weights and biases integration**: Options for tracking experiments using WandB.

4. **Model configuration**: Contains the settings for the model's optimization and network architecture.

5. **Linear parameterization**: Specifies the features used for linear parameterization.


.. code-block:: yaml

   # training
    train:
      eval_freq: 1000 # Frequency (in epochs) for evaluating the model
      batch_size: 250 # Number of samples per gradient update
      epochs: 1000 # Total number of epochs for training
      save_locally: False # If True, plots will be saved locally.

    metrics:
      wasserstein_error: 1 # Order of the Wasserstein error to be computed (order 1 is EMD)
      w_one_ahead: True # If True, calculate the Wasserstein one-ahead error
      w_cumulative: False # If True, calculate the Wasserstein cumulative error (all-steps-ahead)

    #WandB
    wandb:
      save_plots: False # If True, plots will be saved in wandb
      save_model: False # If True, model will be saved in wandb

    # models
    energy:
      # optimization
      optim:
        weight_decay: 0.0
        optimizer: Adam # Choice of optimizer for updating model parameters
        lr: 0.001 # Learning rate for the optimizer
        beta1: 0.9 # Coefficient for computing running averages of gradient
        beta2: 0.999 # Coefficient for computing running averages of squared gradient
        eps: 0.00000001 # Term added to improve numerical stability
        grad_clip: 10.0 # Gradient clipping threshold

      # model architecture
      model:
        layers: [64, 64] # Number of units in each layer of the neural network

      # for linear parameterization
      linear:
        reg: 0.01 # Regularization term for the linear parameterization
        features:
          polynomials:
            degree: 4 # Degree of polynomial features
            sines: False # Enable mixing with sine functions (False to disable)
            cosines: False # Enable mixing with cosine functions (False to disable)
          rbfs:
            n_centers_per_dim: 10 # Number of radial basis function centers per dimension
            domain: [-4, 4] # Domain for radial basis functions
            sigma: 0.5 # Spread (sigma) for radial basis functions
            # types of rbfs to include
            types: [
              # 'linear',
              # 'thin_plate_spline',
              # 'cubic',
              # 'quintic',
              'const',
              # 'multiquadric',
              # 'inverse_multiquadric',
              # 'inverse_quadratic'
            ]
