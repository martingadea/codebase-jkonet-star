Tutorial
============

Generating the data 🧩
----------------------

.. code-block:: bash

   python data_generator.py --potential $potential --interaction $interaction --internal wiener --beta $beta --interaction $interaction

Example 1:
~~~~~~~~~~~

.. code-block:: bash

   python data_generator.py --potential styblinski_tang

Parameters
~~~~~~~~~~~

The `data_generator.py` script accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 50 10 10

   * - Argument
     - Description
     - Type
     - Default
   * - `--load-from-file`
     - Path to a file to load the trajectory from, instead of generating new data.
     - string
     - None
   * - `--potential`
     - Name of the potential energy to use.
     - string
     - 'none'
   * - `--n-timesteps`
     - Number of timesteps for the SDE simulation.
     - int
     - 5
   * - `--dt`
     - Time step size in the SDE simulation.
     - float
     - 0.01
   * - `--internal`
     - Name of the internal energy to use, e.g., 'wiener'.
     - string
     - 'none'
   * - `--beta`
     - Standard deviation of the Wiener process, required if `--internal` is 'wiener'.
     - float
     - 0.0
   * - `--interaction`
     - Name of the interaction energy to use.
     - string
     - 'none'
   * - `--dimension`
     - Dimensionality of the system.
     - int
     - 2
   * - `--n-particles`
     - Number of particles in the system.
     - int
     - 1000
   * - `--batch-size`
     - Batch size for computing couplings.
     - int
     - 1000
   * - `--n-gmm-components`
     - Number of components of the Gaussian Mixture Model. Set to 0 for no GMM.
     - int
     - 10
   * - `--seed`
     - Seed for the random number generator to ensure reproducibility.
     - int
     - 0

Using Custom Data
~~~~~~~~~~~

You can use custom data by loading snapshots from a file using the `--load-from-file` parameter. The snapshots should be in the form of a `(T, N, dim)` array. In this case, the script computes the couplings and fits the densities, but it does not generate new data.

Functions
~~~~~~~~~~~

Available options for `$potential` and `$interaction`:

- double_exp
- styblinski_tang
- rotational
- relu
- flat
- beale
- friedman
- moon
- ishigami
- three_hump_camel
- bohachevsky
- holder_table
- cross_in_tray
- oakley_ohagan
- sphere

Generating All Data for the Paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following script generates all data for the paper:

.. code-block:: bash

   for potential in double_exp styblinski_tang rotational relu flat beale friedman moon ishigami three_hump_camel bohachevsky holder_table cross_in_tray oakley_ohagan sphere
   do
       for beta in 0.0 0.1 0.2 0.5 1.0
       do
           for interaction in double_exp styblinski_tang rotational relu flat beale friedman moon ishigami three_hump_camel bohachevsky holder_table cross_in_tray oakley_ohagan sphere
           do
               python data_generator.py --potential $potential --interaction $interaction --internal wiener --beta $beta
           done
       done

       for dim in 10 20 30 40 50
       do
           for nparticles in 1000 2500 5000 75000 10000
           do
               python data_generator.py --potential $potential --internal wiener --beta 0.0 --n-particles $nparticles --dimension $dim
           done
       done
   done

Note: This script will take significant time and disk space, as it generates a large dataset. We recommend starting with the single experiments of interest as described in Example 1.


Training 🚀
----------

After generating data, you can train a model using the following command:

.. code-block:: bash

   python train.py --solver $solver --dataset $dataset

Where `$solver` can be one of the following:

- jkonet
- jkonet-vanilla
- jkonet-monge-gap
- jkonet-star
- jkonet-star-potential
- jkonet-star-potential-internal
- jkonet-star-linear
- jkonet-star-linear-potential
- jkonet-star-linear-potential-internal

Example 1:
~~~~~~~~~~~

.. code-block:: bash

   python train.py --solver jkonet-star-potential --dataset potential_styblinski_tang_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0

Training All Models on All Data for the Paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following script trains all models on all the data generated:

.. code-block:: bash

   for potential in double_exp styblinski_tang rotational relu flat beale friedman moon ishigami three_hump_camel bohachevsky holder_table cross_in_tray oakley_ohagan sphere
   do
       for beta in 0.0 0.1 0.2 0.5 1.0
       do
           for interaction in double_exp styblinski_tang rotational relu flat beale friedman moon ishigami three_hump_camel bohachevsky holder_table cross_in_tray oakley_ohagan sphere
           do
               for model in jkonet jkonet-vanilla jkonet-monge-gap jkonet-star jkonet-star-potential jkonet-star-potential-internal jkonet-star-linear jkonet-star-linear-potential jkonet-star-linear-potential-internal
               do
                   python train.py --solver $model --dataset potential_$potential\_internal_wiener_beta_$beta\_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0
               done
           done
       done

       for dim in 10 20 30 40 50
       do
           for nparticles in 1000 2500 5000 75000 10000
           do
               python train.py --solver $model --dataset potential_$potential\_internal_wiener_beta_0.0_interaction_none_dt_0.01_T_5_dim_$dim\_N_$nparticles\_gmm_10_seed_0
           done
       done
   done

Note: This script will take a while and consume significant compute resources. The `jkonet` family, in particular, will require days of computation. We recommend starting with individual experiments as described in Example 1. Consider combining this script with data generation and using the `--wandb` flag.

Citation 🙏
----------

If you use this code in your research, please cite our paper:

.. code-block:: latex

   @article{terpin2024learning,
     title={{Learning Diffusion at Lightspeed}},
     author={Terpin, Antonio and Lanzetti, Nicolas and D\"orfler, Florian},
     journal={},
     year={2024},
   }

Contact and Contributing
-------------------------

If you have any questions or would like to contribute to the project, feel free to reach out to [Antonio Terpin](mailto:aterpin@ethz.ch).