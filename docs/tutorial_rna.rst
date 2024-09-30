Tutorial RNA
============


Generating the data 🧩
~~~~~~~~~~~~~~~~~~~~~~~

The dataset that will be used is the one from Moon :cite:`RNA_dataset`. The dataset is publicly available as
``scRNAseq.zip`` at Mendeley Datasets at `this link <https://data.mendeley.com/datasets/v6n743h5ng/>`_.
This dataset tracks the differentiation of human embryonic stem cells over a 27-day period, with cell snapshots
collected at the following time intervals: :math:`t_{0}`: day 0 to 3, :math:`t_{1}`: day 6 to 9, :math:`t_{2}`:
day 12 to 15, :math:`t_{3}`: day 18 to 21, and :math:`t_{4}`: day 24 to 27.

We follow the data pre-processing in :cite:`tong2020trajectorynet` and :cite:`tong2023improving`; in particular,
we use the same processed artifacts of the embryoid data provided in their work, which contains the first 100
components of the principal components analysis (PCA) of the data.
The data is located in "/data/TrajectoryNet/eb_velocity_v5_npz".

The purpose of first script is to load the data and save it in the format we want. Furthermore it allows us to select
the number of components we want to retain as well as the option to crop the data and select just one specific time step.
The data of the trajectories is unbalanced, that is to say that the number of cells is not constant in each time step.
Nevertheless, this is not a problem for the couplings calculations.

.. code-block:: bash

   python preprocess_rna_seq-py --n-components 5

These next script allows us to select the amount of train-test split we want. We also have the option to
specify 0 and
not perform any train-test split.

.. code-block:: bash

   python data_generator.py --load-from-file RNA_PCA_5 --test-split 0.4

Training
~~~~~~~~~~

Finally, in order to train and evaluate the model we run the last script. The argument --eval gives the option to
calculate the evaluation metrics with either the data used for training, or the data left out for evaluation.

.. code-block:: bash

   python train.py --dataset RNA_PCA_5 --solver jkonet-star-time-potential --eval test_data



.. bibliography:: bibliography.bib
   :style: unsrt

