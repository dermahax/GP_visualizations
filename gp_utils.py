"""Shared GP posterior utilities used across notebooks."""

import numpy as np

from kernels import squared_exponential


def gp_posterior(X, y, X_star, kernel_fn, sn, jitter=1e-10):
    """Compute GP posterior mean and covariance for an arbitrary kernel function."""
    X = np.asarray(X)
    y = np.asarray(y)
    X_star = np.asarray(X_star)

    if len(X) == 0:
        Kss = kernel_fn(X_star, X_star)
        return np.zeros(len(X_star)), Kss

    K = kernel_fn(X, X)
    Ks = kernel_fn(X, X_star)
    Kss = kernel_fn(X_star, X_star)

    Ky = K + sn ** 2 * np.eye(len(X))
    L = np.linalg.cholesky(Ky + jitter * np.eye(len(X)))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    cov = Kss - v.T @ v

    return mu, cov


def gp_posterior_se(X, y, X_star, ell, sf, sn, jitter=1e-10):
    """Convenience wrapper for SE-kernel posterior, matching earlier notebook APIs."""

    def se_kernel(a, b):
        return squared_exponential(a, b, lengthscale=ell, variance=sf ** 2)

    return gp_posterior(X, y, X_star, se_kernel, sn, jitter=jitter)


def build_progressive_posterior_frames(
    X_train,
    y_train,
    X_plot,
    kernel_fn,
    sn,
    num_points=None,
):
    """Build posterior frame tuples as points are added one by one."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if num_points is None:
        num_points = len(X_train)

    frames = []
    for n in range(1, num_points + 1):
        X_n = X_train[:n]
        y_n = y_train[:n]
        mu_n, cov_n = gp_posterior(X_n, y_n, X_plot, kernel_fn, sn)
        sd_n = np.sqrt(np.diag(cov_n))
        frames.append((n, X_n, y_n, mu_n, sd_n))

    return frames


def compute_posterior_ylim_from_frames(frames, sn, sigma=2.0, pad_ratio=0.1):
    """Compute stable y-axis limits from a list of posterior frame tuples."""
    y_min = np.inf
    y_max = -np.inf

    for _, X_n, y_n, mu_n, sd_n in frames:
        y_min = min(y_min, np.min(mu_n - sigma * sd_n), np.min(y_n) - sn)
        y_max = max(y_max, np.max(mu_n + sigma * sd_n), np.max(y_n) + sn)

    margin = pad_ratio * (y_max - y_min)
    return y_min - margin, y_max + margin
