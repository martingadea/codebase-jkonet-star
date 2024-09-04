from jax import grad, vmap, numpy as jnp

# RBFs
def rbf_linear(x, c):
    """
    Computes the linear radial basis function (RBF).

    This RBF is simply the negative Euclidean distance between the input `x`
    and the center `c`.

    .. math::
        \mathrm{RBF}(x, c) = -\|x - c\|


    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the linear RBF.
    """
    return - jnp.linalg.norm((x - c))

def rbf_thin_plate_spline(x, c):
    """
    Computes the thin plate spline radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = \|x - c\|^2 \log(\|x - c\| + \epsilon)

    where :math:`\epsilon` is a small constant to avoid numerical issues when
    :math:`\|x - c\|` is close to zero.

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the thin plate spline RBF.
    """
    r = jnp.linalg.norm(x - c)
    return r ** 2 * jnp.log(r + 1e-6)

def rbf_cubic(x, c):
    """
    Computes the cubic radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = \sum_{i} (x_i - c_i)^3

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the cubic RBF.
    """
    return jnp.sum((x - c) ** 3)

def rbf_quintic(x, c):
    """
    Computes the quintic radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = -\sum_{i} (x_i - c_i)^5

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the quintic RBF.
    """
    return -jnp.sum((x - c) ** 5)

def rbf_multiquadric(x, c):
    """
    Computes the multiquadric radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = -\sqrt{\sum_{i} (x_i - c_i)^2 + 1}


    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the multiquadric RBF.
    """
    return -jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_multiquadric(x, c):
    """
    Computes the inverse multiquadric radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = \frac{1}{\sqrt{\sum_{i} (x_i - c_i)^2 + 1}}

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the inverse multiquadric RBF.
    """
    return 1 / jnp.sqrt(jnp.sum((x - c) ** 2) + 1)

def rbf_inverse_quadratic(x, c):
    """
    Computes the inverse quadratic radial basis function (RBF).

    .. math::
        \mathrm{RBF}(x, c) = \frac{1}{\sum_{i} (x_i - c_i)^2 + 1}

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center of the RBF.

    Returns:
        jnp.ndarray: The result of the inverse quadratic RBF.
    """
    return 1 / (jnp.sum((x - c) ** 2) + 1)

def const(x, c):
    """
    Computes the constant function.

    This function always returns 1, regardless of the input `x` or the center `c`.

    .. math::
        \mathrm{const}(x, c) = 1

    Args:
        x (jnp.ndarray): Input data point.
        c (jnp.ndarray): Center (unused).

    Returns:
        jnp.ndarray: The constant value 1.
    """
    return 1
    
rbfs = {
    'linear': rbf_linear,
    'thin_plate_spline': rbf_thin_plate_spline,
    'cubic': rbf_cubic,
    'quintic': rbf_quintic,
    'multiquadric': rbf_multiquadric,
    'inverse_multiquadric': rbf_inverse_multiquadric,
    'inverse_quadratic': rbf_inverse_quadratic,
    'const': const
}