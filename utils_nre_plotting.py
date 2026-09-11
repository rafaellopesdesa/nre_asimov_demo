"""Figures for training, Asimov scans, and discovery toys."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, component in zip(axes, ("S", "B")):
        for member, rows in history[history.component == component].groupby("member"):
            ax.plot(rows.epoch, rows.train_loss, color="C0", alpha=0.5,
                    label="Training" if member == 0 else None)
            ax.plot(rows.epoch, rows.validation_loss, "--", color="C3", alpha=0.5,
                    label="Validation" if member == 0 else None)
        ax.set(xlabel="Epoch", ylabel="BCE", title=f"{component}/reference")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_asimov_scans(results, sizes, scan_mu, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for result, M in zip(results, sizes):
        ax.plot(scan_mu, result["t_mu_A"], label=fr"$M={M:,}$")
    ax.set(xlabel=r"$\mu$", ylabel=r"$t_{\mu,A}$", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_q0(model_toys, simulator_toys, predictions, labels, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    upper = 1.04 * np.quantile(np.r_[model_toys["q0"], simulator_toys["q0"]], 0.999)
    edges = np.linspace(0, upper, 46)
    counts = np.histogram(model_toys["q0"], bins=edges)[0]
    ax.stairs(counts / len(model_toys["q0"]), edges, color="0.5", label="NRE-model toys")
    counts = np.histogram(simulator_toys["q0"], bins=edges)[0]
    ax.errorbar((edges[:-1] + edges[1:]) / 2, counts / len(simulator_toys["q0"]),
                yerr=np.sqrt(counts) / len(simulator_toys["q0"]), fmt="k.", label="Simulator toys")
    for prediction, label in zip(predictions, labels):
        # An interior Asimov fit is needed for this alternative-Wald prediction.
        if prediction["mu_hat"] > 0:
            cdf = norm.cdf(np.sqrt(edges) - np.sqrt(prediction["q0_A"]))
            cdf[0] = 0.0  # Include the point mass at q_0 = 0 in the first bin.
            ax.stairs(np.diff(cdf), edges, label=label)
        else:
            ax.plot([], [], color="none", label=label + " (boundary)")
    ax.set(xlabel=r"$q_0$", ylabel="Probability / bin", yscale="log", title=title,
           ylim=(0.15 / len(model_toys["q0"]), 1))
    ax.legend()
    fig.tight_layout()
    return fig
