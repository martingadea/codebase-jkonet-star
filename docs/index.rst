.. jkonet-star documentation master file, created by
   sphinx-quickstart on Mon Aug 26 11:08:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Learning Diffusion at lightspeed
=================================

.. image:: ../media/cover.png
   :align: left
   :width: 45%
   :alt: Cover

.. image:: ../media/preview.png
   :align: right
   :width: 45%
   :alt: Preview


Diffusion regulates numerous natural processes and the dynamics of many successful generative models. Existing models to learn the diffusion terms from observational data rely on complex bilevel optimization problems and properly model only the drift of the system. We propose a new simple model, JKOnet∗, which bypasses altogether the complexity of existing architectures while presenting significantly enhanced representational capabilities: JKOnet∗
recovers the potential, interaction, and internal energy components of the underlying diffusion process. JKOnet∗ minimizes a simple quadratic loss and drastically outperforms other baselines in terms of sample efficiency, computational complexity, and accuracy. Additionally, JKOnet∗ provides a closed-form optimal solution for linearly
parametrized functionals, and, when applied to predict the evolution of cellular processes, it achieves state-of-the-art accuracy at a fraction of the computational cost of all existing methods. Our methodology is based on the interpretation of diffusion processes as energy-minimizing trajectories in the probability space via the
so-called JKO scheme, which we study via its first-order optimality conditions.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   developer_resources
   applications


Citation 🙏
------------

If you use this code in your research, please cite our paper:

.. code-block:: latex

   @article{terpin2024learning,
     title={{Learning Diffusion at Lightspeed}},
     author={Terpin, Antonio and Lanzetti, Nicolas and Gadea, Martín and D\"orfler, Florian},
     journal={},
     year={2024},
   }

Contact and Contributing
-------------------------

If you have any questions or would like to contribute to the project, feel free to reach out to [Antonio Terpin](mailto:aterpin@ethz.ch).


