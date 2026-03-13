"""
Shared dataset generation for GP visualizations.

The data is drawn from a GP with a squared-exponential kernel (ℓ=1, σ_f=1)
and additive Gaussian noise (σ_n=0.1).  Using a fixed RandomState(42) ensures
the exact same observations are used in every notebook and script.
"""

import numpy as np


def _squared_exponential(x1, x2, lengthscale=1.0, variance=1.0):
    dists = np.subtract.outer(x1, x2) ** 2
    return variance * np.exp(-0.5 * dists / lengthscale ** 2)


def generate_gp_plots_data():
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
    rs = np.random.RandomState(42)

    N = 12
    X_train = rs.uniform(-5, 5, size=N)
    X_train.sort()

    ell_true = 1.0
    sf_true = 1.0
    sn_true = 0.1

    K = _squared_exponential(
        X_train, X_train, lengthscale=ell_true, variance=sf_true ** 2
    )
    f = rs.multivariate_normal(np.zeros(N), K)
    y = f + rs.normal(0, sn_true, size=N)

    return X_train, y, sn_true
