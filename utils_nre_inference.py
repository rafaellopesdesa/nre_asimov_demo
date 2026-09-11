"""Finite Asimov samples, likelihood scans, and Poisson toys."""

import numpy as np


def h(r, mu, lambda_s, Z):
    return (r / Z) @ (lambda_s * np.array([mu, 1.0]))


def signal_to_background(r, lambda_s, Z):
    return (lambda_s[0] * r[:, 0] / Z[0]) / (lambda_s[1] * r[:, 1] / Z[1])


def log_likelihood(mu, ratio, w, lambda_S):
    """Return ell(mu) - ell(0)."""
    mu = np.asarray(mu)
    return -mu * lambda_S + np.sum(w * np.log1p(mu[..., None] * ratio), axis=-1)


def fit(ratio, w, lambda_S):
    """Maximize the concave likelihood on mu >= 0 by score bisection."""
    low = np.zeros(w.shape[:-1])
    high = np.where(np.sum(w * ratio, axis=-1) > lambda_S,
                    np.sum(w, axis=-1) / lambda_S, 0.0)
    for _ in range(64):
        mu = (low + high) / 2
        score = np.sum(w * ratio / (1 + mu[..., None] * ratio), axis=-1) - lambda_S
        low = np.where(score > 0, mu, low)
        high = np.where(score > 0, high, mu)
    mu_hat = (low + high) / 2
    return mu_hat, 2 * log_likelihood(mu_hat, ratio, w, lambda_S)


def asimov_scan(r, w_A, lambda_s, Z, scan_mu):
    ratio = signal_to_background(r, lambda_s, Z)
    mu_hat, q0_A = fit(ratio, w_A, lambda_s[0])
    ell = np.array([log_likelihood(mu, ratio, w_A, lambda_s[0]) for mu in scan_mu])
    return dict(mu_hat=float(mu_hat), q0_A=float(q0_A), t_mu_A=q0_A - 2 * ell)


def finite_reference_asimov(r, lambda_s, mu_A, scan_mu):
    Z = r.mean(axis=0)
    w_A = h(r, mu_A, lambda_s, Z) / len(r)
    return asimov_scan(r, w_A, lambda_s, Z, scan_mu)


def simulator_asimov(r_S, r_B, lambda_s, Z, mu_A, scan_mu):
    r = np.concatenate([r_S, r_B])
    w_A = np.r_[np.full(len(r_S), mu_A * lambda_s[0] / len(r_S)),
                np.full(len(r_B), lambda_s[1] / len(r_B))]
    return asimov_scan(r, w_A, lambda_s, Z, scan_mu)


def compress_reference(r, lambda_s, Z, n_bins):
    """Integrate the finite model into bins of log(signal/background)."""
    log_ratio = np.log(signal_to_background(r, lambda_s, Z))
    quantiles = np.quantile(log_ratio, np.linspace(0, 1, n_bins // 4 + 1)[1:-1])
    uniform = np.linspace(log_ratio.min(), log_ratio.max(), 3 * n_bins // 4 + 2)[1:-1]
    edges = np.unique(np.r_[-np.inf, quantiles, uniform, np.inf])
    counts = np.histogram(log_ratio, bins=edges)[0]
    occupied = np.flatnonzero(counts)
    edges = np.r_[-np.inf, edges[occupied[1:]], np.inf]
    probabilities = np.array([
        np.histogram(log_ratio, bins=edges, weights=r[:, s])[0] / r[:, s].sum()
        for s in range(2)
    ])
    ratio = lambda_s[0] * probabilities[0] / (lambda_s[1] * probabilities[1])
    return edges, probabilities, ratio


def simulator_probabilities(banks, lambda_s, Z, edges):
    return np.array([
        np.histogram(np.log(signal_to_background(r, lambda_s, Z)), bins=edges)[0] / len(r)
        for r in banks
    ])


def run_toys(ratio, probabilities, lambda_s, mu_A, n_toys, seed, batch_size=128):
    rng = np.random.default_rng(seed)
    rates = mu_A * lambda_s[0] * probabilities[0] + lambda_s[1] * probabilities[1]
    mu_hat, q0 = np.empty(n_toys), np.empty(n_toys)
    for start in range(0, n_toys, batch_size):
        stop = min(start + batch_size, n_toys)
        counts = rng.poisson(rates, size=(stop - start, len(rates)))
        mu_hat[start:stop], q0[start:stop] = fit(ratio, counts, lambda_s[0])
    return dict(mu_hat=mu_hat, q0=q0)
