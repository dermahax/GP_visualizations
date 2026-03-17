"""Shared dataset generation for GP visualizations.

All datasets are sampled from GP priors with additive Gaussian noise.
Kernel implementations are centralized in kernels.py so notebooks and scripts
use identical kernel definitions.
"""

import numpy as np

from kernels import linear_kernel, periodic_kernel, squared_exponential


def generate_gp_plots_data(seed=42):
    """Return the shared synthetic GP dataset.

    Returns
    -------
    X_train : ndarray, shape (12,)
        Sorted input locations sampled uniformly from [-5, 5].
    y : ndarray, shape (12,)
        Noisy observations: y = f(x) + ε,  ε ~ N(0, sn²).
    sn : float
        Observation noise standard deviation (0.1).
    """
    rs = np.random.RandomState(seed)

    N = 12
    X_train = rs.uniform(-5, 5, size=N)
    X_train.sort()

    ell_true = 1.0
    sf_true = 1.0
    sn_true = 0.1

    K = squared_exponential(
        X_train, X_train, lengthscale=ell_true, variance=sf_true ** 2
    )
    f = rs.multivariate_normal(np.zeros(N), K)
    y = f + rs.normal(0, sn_true, size=N)

    return X_train, y, sn_true


def generate_gp_kernel_addition_data(seed=123):
    """Return a 100-point dataset sampled from a composite GP kernel.

    Kernel used:
        linear + periodic + squared-exponential

    Returns
    -------
    X_train : ndarray, shape (100,)
        Sorted input locations sampled uniformly from [-5, 5].
    y : ndarray, shape (100,)
        Noisy observations: y = f(x) + ε,  ε ~ N(0, sn²).
    sn : float
        Observation noise standard deviation (0.1).
    """
    rs = np.random.RandomState(seed)

    n_points = 100
    X_train = rs.uniform(-5, 5, size=n_points)
    X_train.sort()

    sn_true = 0.1

    K_linear = linear_kernel(
        X_train,
        X_train,
        slope_variance=0.25,
        bias_variance=0.8,
    )
    K_periodic = periodic_kernel(
        X_train,
        X_train,
        lengthscale=1.0,
        variance=0.9,
        period=2.0,
    )
    K_se = squared_exponential(
        X_train,
        X_train,
        lengthscale=1.2,
        variance=0.7,
    )

    K = K_linear + K_periodic + K_se

    f = rs.multivariate_normal(
        np.zeros(n_points), K + 1e-10 * np.eye(n_points))
    y = f + rs.normal(0, sn_true, size=n_points)

    return X_train, y, sn_true
