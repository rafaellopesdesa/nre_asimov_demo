"""Train the fixed selection and estimate the selected yields."""

import numpy as np

from utils_nre import simulate, predict, train_classifier, fit_scaler


def train_selection(n_train, n_validation, config, device, seed):
    rng = np.random.default_rng(seed)
    training = [simulate(s, n_train, rng) for s in ("S", "B")]
    validation = [simulate(s, n_validation, rng) for s in ("S", "B")]
    return train_classifier(*training, *validation, **config, device=device,
                            seed=seed, scaler=fit_scaler(training))


def calibrate_cut(model, n, inclusive_yields, target, seed):
    rng = np.random.default_rng(seed)
    edges = np.r_[-np.inf, np.linspace(-20, 20, 4001), np.inf]
    histograms = []
    for component in ("S", "B"):
        counts = np.zeros(len(edges) - 1)
        for start in range(0, n, 100_000):
            x = simulate(component, min(100_000, n - start), rng)
            counts += np.histogram(predict(model, x), bins=edges)[0]
        histograms.append(counts)
    accepted = np.cumsum(np.array(histograms)[:, ::-1], axis=1)[:, ::-1]
    signal, background = inclusive_yields[:, None] * accepted / n
    index = np.flatnonzero((signal > 0) & (background <= target * signal))[0]
    return float(edges[index])


def selected_yields(selection, n, inclusive_yields, seed):
    rng = np.random.default_rng(seed)
    counts = []
    for component in ("S", "B"):
        accepted = 0
        for start in range(0, n, 100_000):
            x = simulate(component, min(100_000, n - start), rng)
            accepted += selection(x).sum()
        counts.append(accepted)
    return inclusive_yields * np.array(counts) / n
