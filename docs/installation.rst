Installation
============

Installation with Docker
--------------------------

Docker (https://www.docker.com/) should be installed on your machine.

To build the Docker image, make sure Docker is running and run the following command from the root of the repository:

.. code-block:: bash

    docker build -t jkonet-star-app .

Here an example on how to generate some data and train using the docker image

.. code-block::

    docker run jkonet-star-app python data_generator.py --potential styblinski_tang
    docker run jkonet-star-app python train.py --solver jkonet-star-potential --dataset potential_styblinski_tang_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0_split_0

Installation in OS and Ubuntu
------------------------------
The following works on macOS 13.2.1 and should work also on Ubuntu.

1. Install miniconda.
2. Create an environment:

.. code-block:: bash

    conda create --name jkonet-star python=3.12
    conda activate jkonet-star

3. Install requirements

.. code-block:: bash

    pip install -r requirements.txt

Installation in Windows
------------------------

The following works on Windows 11. For compatibility issues, the Python version required is the 3.9.

1. Install miniconda.
2. Create an environment:

.. code-block:: bash

    conda create --name jkonet-star python=3.9
    conda activate jkonet-star

3. Install requirements

.. code-block:: bash

    pip install -r requirements-win.txt




