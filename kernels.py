"""Shared kernel functions for GP visualizations."""

import numpy as np


def squared_exponential(x1, x2, lengthscale=1.0, variance=1.0):
    """Squared-exponential (RBF) kernel."""
    dists = np.subtract.outer(np.asarray(x1), np.asarray(x2)) ** 2
    return variance * np.exp(-0.5 * dists / lengthscale ** 2)


def linear_kernel(x1, x2, slope_variance=1.0, bias_variance=1.0):
    """Linear kernel: k(x, x') = bias_variance + slope_variance * x * x'."""
    return bias_variance + slope_variance * np.outer(np.asarray(x1), np.asarray(x2))


def periodic_kernel(x1, x2, lengthscale=1.0, variance=1.0, period=2.0):
    """Periodic kernel with period p and smoothness controlled by lengthscale."""
    dists = np.abs(np.subtract.outer(np.asarray(x1), np.asarray(x2)))
    return variance * np.exp(
        -2.0 * np.sin(np.pi * dists / period) ** 2 / lengthscale ** 2
    )
