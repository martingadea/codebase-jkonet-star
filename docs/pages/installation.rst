Installation guide
==================

Installation via Docker
-----------------------

Before proceeding, ensure Docker is installed on your machine. You can download Docker from the official site: `https://www.docker.com/ <https://www.docker.com/>`_.

Once Docker is installed and running, follow these steps to build the Docker image. Execute the following command from the root directory of the repository:

.. code-block:: bash

    docker build -t jkonet-star-app .

Running JKOnet\* using Docker
------------------------------------

After building the image, you can generate data and train models by executing the following commands. Below is an example of how to use the Docker image for these tasks:

.. code-block:: bash

    # Generate data using the Styblinski-Tang potential
    docker run jkonet-star-app python data_generator.py --potential styblinski_tang

    # Train the model using the generated dataset
    docker run jkonet-star-app python train.py --solver jkonet-star-potential --dataset potential_styblinski_tang_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0_split_0

Installation on macOS and Ubuntu
--------------------------------

These steps have been tested on macOS 13.2.1 and should also work on Ubuntu systems.

Steps:

1. **Install Miniconda**

   Download and install Miniconda from the official website: `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.

2. **Create a Conda environment**

   Open a terminal and run the following commands to create and activate a new Conda environment:

   .. code-block:: bash

       conda create --name jkonet-star python=3.12
       conda activate jkonet-star

3. **Install the required packages**

   Once the environment is activated, install the necessary dependencies:

   .. code-block:: bash

       pip install -r requirements.txt

Installation on Windows
-----------------------

The following instructions are for Windows 11 users. Please note that Python 3.9 is required for compatibility.

Steps:

1. **Install Miniconda**

   Download and install Miniconda from the official website: `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.

2. **Create a Conda environment**

   Run the following commands in your terminal to create and activate the environment with Python 3.9:

   .. code-block:: bash

       conda create --name jkonet-star python=3.9
       conda activate jkonet-star

3. **Install the required packages**

   Once the environment is activated, install the necessary dependencies for Windows:

   .. code-block:: bash

       pip install -r requirements-win.txt
