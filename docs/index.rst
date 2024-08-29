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

Diffusion regulates a phenomenal number of natural processes and the dynamics of
many successful generative models. Existing models to learn the diffusion terms from
observational data rely on complex bilevel optimization problems and properly model
only the drift of the system. We propose a new simple model, JKOnet, which bypasses
altogether the complexity of existing architectures while presenting significantly
enhanced representational capacity: JKOnet recovers the potential, interaction, and
internal energy components of the underlying diffusion process. JKOnet minimizes a
simple quadratic loss, runs at lightspeed, and drastically outperforms other baselines
in practice. Additionally, JKOnet provides a closed-form optimal solution for linearly
parametrized functionals. Our methodology is based on the interpretation of diffusion
processes as energy-minimizing trajectories in the probability space via the so-called
JKO scheme, which we study via its first-order optimality conditions, in light of
few-weeks-old advancements in optimization in the probability space.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   api


