Benchmarks 🔥
====================

In this page we report the benchmarks for the JKOnet\* model on the synthetic data. For the results related to the single-cell data, please refer to the :doc:`tutorial_rna` page. Check also the `paper <https://arxiv.org/abs/2406.12616>`_ for more details.

TODO FIX THE TEXT BELOW AND IMPORT IMAGES

Experiment 4.1: thin_plate_spline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python data_generator.py --potential thin_plate_spline


Post-processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python post_processing.py --potential thin_plate_spline


Note


Experiment 4.2: cubic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python data_generator.py --potential cubic

  
Post-processing

.. code-block:: bash

   python post_processing.py --potential cubic



TODO note we did not test the bash scripts in Docker yet, but you can reproduce the results by running the commands in the terminal.