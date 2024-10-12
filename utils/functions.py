"""
This module provides a collection of energy landscape functions commonly used for optimization, testing, and benchmarking purposes. These functions are intentionally not vectorized to enable the use of ``jax.grad`` for automatic differentiation. For automatic vectorization, you can use ``jax.vmap``.

The functions in this module represent a variety of optimization landscapes, including convex, non-convex, and complex synthetic functions. They are commonly used in sensitivity analysis, regression tasks, and testing optimization algorithms.

Available Functions:
---------------------
- ``styblinski_tang``: A non-convex function used to test optimization algorithms.
- ``holder_table``: A complex, non-convex optimization function.
- ``cross_in_tray``: Another non-convex function, used for benchmarking optimization algorithms.
- ``oakley_ohagan``: A synthetic function used for testing.
- ``moon``: A complex function that uses an interaction matrix for polynomial expansions.
- ``ishigami``: A function used in sensitivity analysis.
- ``friedman``: A function used in regression and sensitivity analysis.
- ``sphere``: A simple convex function for optimization testing.
- ``bohachevsky``: A non-convex function with trigonometric terms.
- ``three_hump_camel``: A non-convex optimization function.
- ``beale``: A well-known non-convex function used for optimization testing.
- ``double_exp``: A double exponential function used in optimization problems.
- ``relu``: A rectified linear unit (ReLU) function.
- ``rotational``: A trigonometric-based optimization function.
- ``flat``: A trivial function that returns zero, useful for testing.

Example Usage:
--------------
To use any of the provided functions, pass a ``jax.numpy`` array as input:

.. code-block:: python

    import jax.numpy as jnp
    from module_name import potentials_all

    v = jnp.array([1.0, 2.0, 3.0])
    result = potentials_all['styblinski_tang'](v)

You can also compute the gradient of these functions using `jax.grad`:

.. code-block:: python

    from jax import grad
    gradient = grad(potentials_all['styblinski_tang'])(v)

Note:
-----
For vectorized operations, you can use `jax.vmap` over the provided functions.
"""

import jax.numpy as jnp

def styblinski_tang(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Styblinski-Tang function.

    .. math::
        f(v) = 0.5 \sum_{i=1}^{d} (v_i^4 - 16v_i^2 + 5v_i)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Styblinski-Tang function.
    """
    u = jnp.square(v)
    return 0.5 * jnp.sum(jnp.square(u) - 16 * u + 5 * v)

def holder_table(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Holder Table function.

    .. math::
        f(v) = -\\left|\sin(v_1)\cos(v_2)\exp\\left(\\left|1 - \\frac{\sqrt{v_1^2 + v_2^2}}{\pi}\\right|\\right)\\right|

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Holder Table function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * jnp.abs(jnp.sin(v1) * jnp.cos(v2) * jnp.exp(jnp.abs(1 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi)))

def cross_in_tray(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Cross-in-Tray function.

    .. math::
        f(v) = -2 \\left(\\left| \sin(z_1) \sin(z_2) \exp\\left( \\left| 10 - \\frac{||v||}{\pi} \\right| \\right) \\right| + 1 \\right)^{0.1}

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Cross-in-Tray function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return -2 * (jnp.abs(jnp.sin(v1) * jnp.sin(v2) * jnp.exp(jnp.abs(10 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi))) + 1)**0.1

def oakley_ohagan(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Oakley-Ohagan function.

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Oakley-Ohagan function.

    .. math::
        f(v) = 5 \sum_{i=1}^{d} (\sin(v_i) + \cos(v_i) + v_i^2 + v_i)
    """
    return 5 * jnp.sum(jnp.sin(v) + jnp.cos(v) + jnp.square(v) + v)

def moon(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Moon function.

    .. math::
        f(v) = \max\\left(-100, \min\\left(100, v^\\top A v\\right)\\right)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Moon function, clipped between -100 and 100.
    """
    interaction_matrix = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2.08, 1.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.11, 2.18, -1.70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.76, 0.58, 0.84, 1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.57, -1.21, 1.20, -0.49, -3.23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.72, -7.15, -2.35, 1.74, 2.75, -1.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.47, -1.29, -0.16, 1.29, -1.40, 2.34, 0.21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.39, -0.19, -0.35, 0.24, -3.90, -0.03, -4.16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.40, -2.75, -5.93, -4.73, -0.70, -0.80, -0.37, 0.26, -1.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.09, -1.16, -1.15, 3.27, -0.17, 0.13, -1.27, -0.30, 0.77, 3.06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.7, -1.09, 1.89, 1.87, -3.38, -3.97, 2.78, -2.69, 1.09, 2.46, 3.34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.27, 0.89, -3.47, 1.42, -1.87, 1.99, 1.37, -2.56, -1.15, 5.80, 2.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.03, -0.16, -0.07, -0.96, -0.17, 0.45, -2.75, 28.99, -1.09, -5.15, -1.77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.07, 4.43, 0.60, -0.91, 1.56, 1.77, -3.15, -2.13, -2.74, -2.05, -3.16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.23, 1.65, -1.09, 2.06, 2.40, -0.50, 1.86, 1.36, 1.59, 3.17, 1.89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.46, -1.25, -3.23, 2.89, -1.70, 1.86, 0.12, 1.45, .41, 3.40, 2.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.31, -1.35, 0.44, 0.25, 0.32, 0.02, -0.74, 3.09, 0.48, -0.49, -0.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2.94, 1.15, 1.24, 1.97, 2.11, -2.08, 1.06, -1.73, 2.16, -6.71, -3.78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.63, -19.71, 2.13, 3.04, -0.20, 1.78, -3.76, -1.66, 0.34, -0.74, 0.98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.07, 23.72, -0.71, 2.00, 1.39, 1.76, -0.43, -3.94, 4.17, 2.78, 1.40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.44, 1.42, 1.64, 1.64, -2.01, 1.30, 1.25, -2.56, 0.73, -0.41, -0.59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    v = v / 4
    d = v.shape[0]
    while d < 20:
        v = jnp.concatenate([v, v ** 2], axis=0)
        d = v.shape[0]
    v = jnp.concatenate([jnp.ones(1), v[:20]], axis=0)
    return jnp.clip(jnp.dot(v, jnp.dot(interaction_matrix, v)), -100, 100)

def ishigami(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Ishigami function.

    .. math::
        f(v) = \sin(z_1) + 7 \sin(z_2)^2 + 0.1 \\left(\\frac{z_1 + z_2}{2}\\right)^4 \sin(z_1)

    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Ishigami function.
    """

    d = v.shape[0]
    v0 = jnp.mean(v[:d//2])
    v1 = jnp.mean(v[d//2:])
    v2 = (v0 + v1) / 2
    return jnp.sin(v0) + 7 * jnp.sin(v1) ** 2 + 0.1 * v2 ** 4 * jnp.sin(v0)

def friedman(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Friedman function.

    .. math::
        f(v) = \\frac{1}{100}\\biggl(10\sin\\left(2\pi(z_1 - 7)(z_2 - 7)\\right) + 
        20\\left(2(z_1 - 7)\sin(z_2 - 7)- \\frac{1}{2}\\right)^2 \\\\+
        10\\left(2(z_1 - 7)\cos(z_2 - 7) - 1\\right)^2 + \\frac{1}{10}(z_2 - 7)\sin(2(z_1 - 7))\\biggr)



    Parameters
    ----------
    v : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The result of the Friedman function, scaled down by a factor of 100.
    """
    v = 2 * (v - 7)
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:]) / 2
    v3 = v1 * jnp.sin(v2)
    v4 = v1 * jnp.cos(v2)
    v5 = v2 * jnp.sin(v1)
    return (10 * jnp.sin(jnp.pi * v1 * v2) + 20 * (v3 - 0.5) ** 2 + 10 * (v4 - 1) ** 2 + 0.1 * v5) / 100

def sphere(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Sphere function.

    .. math::
        f(v) = -10||x||^2

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Sphere function.
    """
    return -10 * jnp.sum(jnp.square(v))

def bohachevsky(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Bohachevsky function.

    .. math::
        f(v) = v_1^2 + 2v_2^2 - 0.3 \cos(3 \pi v_1) - 0.4 \cos(4 \pi v_2) + 0.7


    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Bohachevsky function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 10 * (jnp.square(v1) + 2 * jnp.square(v2) - 0.3 * jnp.cos(3 * jnp.pi * v1) - 0.4 * jnp.cos(4 * jnp.pi * v2))

def three_hump_camel(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Three-Hump Camel function.

    .. math::
        f(v) = 2z_1^2 - 1.05z_1^4 + \\frac{z_1^6}{6} + z_1z_2 + z_2^2

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Three-Hump Camel function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return 2 * jnp.square(v1) - 1.05 * jnp.power(v1, 4) + jnp.power(v1, 6) / 6 + v1 * v2 + jnp.square(v2)

def beale(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Beale function.

    .. math::
        f(v) = \\frac{1}{100}((1.5 - z_1 + z_1z_2)^2 + (2.25 - z_1 + z_1z_2)^2 + (2.625 - z_1 + z_1z_2^3)^2)

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Beale function, scaled down by a factor of 100.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    return jnp.clip(
        jnp.square(1.5 - v1 + v1 * v2) + jnp.square(2.25 - v1 + v1 * jnp.square(v2)) + jnp.square(2.625 - v1 + v1 * jnp.power(v2, 3)) / 100, -10, 10)

def double_exp(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Double Exponential function.

    .. math::
        f(v) = 200\exp\\left(-\\frac{||v - m\mathbf{1}||^2}{\sigma}\\right) + \exp\\left(-\\frac{||v + m\mathbf{1}||}{s}\\right)

    where :math:`d = 3` and :math:`s = 20`.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Double Exponential function.
    """
    s = 20
    d = 3
    return 200 * (jnp.exp(-jnp.sum(jnp.square(v - d))/s) + jnp.exp(-jnp.sum(jnp.square(v + d))/s))

def relu(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the ReLU (Rectified Linear Unit) function.

    .. math::
        f(v) = \max(0, v)

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the ReLU function.
    """
    r = -50 * jnp.clip(v, a_min=0)
    if r.ndim > 0:
        return jnp.sum(r)
    return r

def rotational(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Rotational function.

    .. math::
        f(v) = 10 \cdot \\text{ReLU}(\\theta + \pi)

    where :math:`\theta = \\arctan\\left(\frac{v_2 + 5}{v_1 + 5}\\right)`.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Rotational function.
    """
    d = v.shape[0]
    v1 = jnp.mean(v[:d//2])
    v2 = jnp.mean(v[d//2:])
    theta = jnp.arctan2(v2 + 5, v1 + 5)
    return 10 * relu(theta + jnp.pi)

def flat(v: jnp.ndarray) -> jnp.ndarray:
    r"""
    Computes the Flat function.

    Parameters
    ----------
        v (jnp.ndarray): Input array.

    Returns
    -------
        jnp.ndarray: The result of the Flat function (always 0).
    """
    return 0.


potentials_all = {
    'double_exp': double_exp,
    'rotational': rotational,
    'relu': relu,
    'flat': flat,
    'beale': beale,
    'friedman': friedman,
    'moon': moon,
    'ishigami': ishigami,
    'three_hump_camel': three_hump_camel,
    'bohachevsky': bohachevsky,
    'holder_table': holder_table,
    'cross_in_tray': cross_in_tray,
    'oakley_ohagan': oakley_ohagan,
    'sphere': sphere,
    'styblinski_tang': styblinski_tang
}

interactions_all = potentials_all