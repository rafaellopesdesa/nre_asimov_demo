"""Gaussian mixtures and detector response."""

import numpy as np


def build_cov(sigmas, correlations=None):
    """Build a covariance matrix from standard deviations and correlations."""
    sigmas = np.asarray(sigmas, dtype=float)
    corr = np.eye(len(sigmas))
    if correlations:
        for (i, j), rho in correlations.items():
            corr[i, j] = corr[j, i] = rho
    return np.outer(sigmas, sigmas) * corr


BASE_MEAN = np.array([2.5, 2.0, 3.0, 1.5, 2.5])
BASE_SIGMA = np.array([2.5, 2.4, 2.5, 2.4, 2.5])
BASE_COV = build_cov(BASE_SIGMA)
BASE_FRAC = 0.20


def background_components():
    """Return the bimodal, correlated background mixture components."""
    bg1 = (
        0.45,
        np.array([2.0, 1.0, 2.5, 0.8, 2.0]),
        build_cov([1.0, 0.9, 1.2, 0.9, 1.0], {(0, 1): 0.6, (3, 4): 0.5}),
    )
    bg2 = (
        0.35,
        np.array([4.2, 3.1, 4.6, 2.6, 3.6]),
        build_cov([0.9, 1.0, 1.0, 0.9, 1.1], {(0, 1): -0.5, (2, 3): 0.5}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [bg1, bg2, base]


def signal_components(v=10):
    """Return the signal mixture components at parameter value ``v``."""
    t = v / 10.0
    sig_a = (
        0.50,
        np.array([2.2, 1.3, 2.8, 1.0, 2.2])
        + t * np.array([2.3, 1.9, 1.5, 1.3, 1.8]),
        build_cov([0.9, 0.8, 1.0, 0.8, 0.9], {(0, 2): 0.5, (1, 4): 0.4}),
    )
    sig_b = (
        0.30,
        np.array([3.5, 2.5, 3.8, 2.0, 3.0])
        + t * np.array([0.8, 0.7, 1.2, 0.6, 1.0]),
        build_cov([0.8, 0.9, 0.9, 0.8, 1.0], {(0, 1): 0.5, (2, 4): -0.4}),
    )
    base = (BASE_FRAC, BASE_MEAN, BASE_COV)
    return [sig_a, sig_b, base]


def smearing_parameters():
    """Return the nominal detector-response scale and resolution vectors."""
    scale = np.array([1.2, 1.1, 0.99, 0.96, 1.01])
    resolution = np.array([1.0, 0.1, 0.9, 1.3, 0.2])
    return [scale, resolution]
