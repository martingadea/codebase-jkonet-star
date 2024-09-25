Design choices
================

In this section we discuss the mayor design choices along the project.

Mixture of Gaussians for the density
-------------------------------------

If we consider the internal energy term of the energy functional :math:`J` (:math:`\theta_3 \neq 0`),
we must estimate the density :math:`\rho_t` and its gradient :math:`\nabla \rho_t` from the empirical
probability measures :math:`\mu_t`. To estimate :math:`\rho_t`, we use a mixture of 10 Gaussians. This
approach involves representing :math:`\rho_t` as a weighted sum of multiple Gaussian distributions, each
with its own mean and variance. The mixture model allows for a more flexible and accurate approximation
of complex density functions. To compute :math:`\nabla \rho_t`, we take advantage of the automatic
differentiation capabilities provided by :code:`JAX`.

Precomputation of couplings
----------------------------

In this method, the couplings are not treated as optimization variables, allowing them to be precomputed.
However, if the number of couplings is excessively large, this precomputation could lead to significant
memory usage. In such situations, it may be more efficient to calculate the couplings dynamically during
each iteration of the optimization loop.

Linear vs Non-Linear parametrization
-------------------------------------

Linear parametrization offers several advantages, including guarantees in performance, ease of
validation, and often better results when the features are well-chosen. However, it poses challenges
such as significant scaling with dimensionality and the need for carefully selected representative features.
On the other hand, non-linear parametrization, such as in neural networks, simplifies the process by automating
feature selection and tends to handle high-dimensional data more effectively. However, it requires more data
and the results can be less interpretable compared to linear models.

