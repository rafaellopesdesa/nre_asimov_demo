"""Simulation, balanced BCE training, and ratio prediction."""

import numpy as np
import torch
from torch import nn

from utils_distributions import (
    background_components, signal_components, smearing_parameters,
)


def simulate(component, n, rng):
    components = {"S": signal_components, "B": background_components}[component]()
    labels = rng.choice(len(components), n, p=[c[0] for c in components])
    x = np.empty((n, 5))
    for index, (_, mean, covariance) in enumerate(components):
        mask = labels == index
        x[mask] = rng.multivariate_normal(mean, covariance, mask.sum())
    scale, resolution = smearing_parameters()
    return (x * scale + rng.normal(scale=resolution, size=x.shape)).astype(np.float32)


def sample(component, n, selection, rng):
    """Draw from p_S, p_B, or the equal mixture after selection."""
    x = np.empty((n, 5), dtype=np.float32)
    if component == "ref":
        signal = rng.random(n) < 0.5
        x[signal] = sample("S", signal.sum(), selection, rng)
        x[~signal] = sample("B", (~signal).sum(), selection, rng)
    else:
        start = 0
        while start < n:
            candidates = simulate(component, 100_000, rng)
            accepted = candidates[selection(candidates)][:n - start]
            x[start:start + len(accepted)] = accepted
            start += len(accepted)
    return x


class Classifier(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.register_buffer("offset", torch.zeros(5))
        self.register_buffer("scale", torch.ones(5))
        layers = []
        for left, right in zip([5] + widths, widths):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        self.network = nn.Sequential(*layers, nn.Linear(widths[-1], 1))

    def forward(self, x):
        return self.network((x - self.offset) * self.scale - 1.5).squeeze(-1)


class NRE(nn.Module):
    def __init__(self, widths, ensemble_size):
        super().__init__()
        self.models = nn.ModuleList([
            nn.ModuleList([Classifier(widths) for _ in range(ensemble_size)])
            for _ in range(2)
        ])

    def forward(self, x):
        return torch.stack([
            torch.stack([model(x).double().exp() for model in members]).mean(0)
            for members in self.models
        ], dim=1)


@torch.no_grad()
def predict(model, x, batch_size=65_536):
    device = next(model.parameters()).device
    return np.concatenate([
        model(torch.as_tensor(x[start:start + batch_size], device=device))
        .cpu().numpy().astype(np.float64)
        for start in range(0, len(x), batch_size)
    ])


def train_classifier(positive, negative, val_positive, val_negative, *,
                     widths, epochs, batch_size, learning_rate, scheduler_step,
                     scheduler_gamma, device, seed, scaler, patience=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Classifier(widths).to(device)
    model.offset.copy_(torch.as_tensor(scaler[0], device=device))
    model.scale.copy_(torch.as_tensor(scaler[1], device=device))
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    x = torch.from_numpy(np.concatenate([positive, negative]))
    y = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
    x_val = torch.from_numpy(np.concatenate([val_positive, val_negative]))
    y_val = torch.cat([torch.ones(len(val_positive)), torch.zeros(len(val_negative))])
    history, best_loss, stale = [], np.inf, 0
    for epoch in range(epochs):
        lr = max(1e-10, learning_rate * scheduler_gamma ** (epoch // scheduler_step))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = rng.permutation(len(x))
        train_loss = 0.0
        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x[indices].to(device)), y[indices].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(indices) / len(x)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                batch = slice(start, start + batch_size)
                loss = loss_function(model(x_val[batch].to(device)), y_val[batch].to(device))
                val_loss += loss.item() * len(x_val[batch]) / len(x_val)
        history.append(dict(epoch=epoch + 1, train_loss=train_loss,
                            validation_loss=val_loss, learning_rate=lr))
        print(f"Epoch {epoch + 1}/{epochs}: BCE={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.1e}")
        if patience is not None:
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale == patience:
                break
    if patience is not None:
        model.load_state_dict(best_state)
    return model, history


def fit_scaler(banks):
    offset = np.min([x.min(0) for x in banks], axis=0)
    maximum = np.max([x.max(0) for x in banks], axis=0)
    return offset, 3.0 / (maximum - offset)


def train_nre(training, validation, config, ensemble_size, device, seed):
    nre = NRE(config["widths"], ensemble_size).to(device)
    scaler = fit_scaler(list(training.values()))
    histories = []
    for s, component in enumerate(("S", "B")):
        for member in range(ensemble_size):
            print(f"{component}/ref, member {member + 1}")
            model, history = train_classifier(
                training[component], training["ref"], validation[component], validation["ref"],
                **config, device=device, seed=seed + s * 1_000_000 + member * 10_007,
                scaler=scaler,
            )
            nre.models[s][member] = model
            histories.extend(dict(component=component, member=member, **row) for row in history)
    return nre.eval(), histories
