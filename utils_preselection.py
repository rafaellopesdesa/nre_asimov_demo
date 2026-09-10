"""Independent, cached signal/background preselection for the NRE demonstration.

Training, validation, cut calibration, and yield estimation use separate random
streams. Completed classifier, calibration, and yield artifacts are immutable:
changing the yield-bank size never retrains the classifier.
"""

import json
from pathlib import Path

import numpy as np

from utils_nre import (
    FEATURES, CacheError, _device, _fingerprint, _json, _network,
    _new_artifact, _positive_integer, _rng, _simulator_contract, _torch,
    _train_member, _training_config, _verified_manifest, _write_manifest,
    simulate_reconstructed,
)


SELECTION_VERSION = "nre-preselection-v1"
DEFAULT_SELECTION_TRAINING_CONFIG = {
    "ensemble_size": 1, "hidden_layers": 3, "hidden_features": 256,
    "activation": "swish", "epochs": 30, "batch_size": 2048,
    "learning_rate": 1.0e-3, "scheduler_step": 30, "scheduler_gamma": 1.0,
    "minimum_learning_rate": 1.0e-10, "patience": 8,
    "checkpoint_selection": "best_validation", "device": "auto",
    "prediction_batch_size": 8192,
}
_TRAIN_GENERATION_BATCH_SIZE = 100_000
_MINIMUM_CALIBRATION_ACCEPTS = 50
_LOG_ODDS_EDGES = np.linspace(-20.0, 20.0, 4001)


class Selection:
    """Frozen classifier cut and selected physical yields, ordered (S, B)."""

    def __init__(self, model, scaler, config, device, history):
        self.model = model
        self.offset, self.scale = scaler
        self.config = config
        self.device = device
        self.history = history
        self.ratio_cut = None
        self.yields = None
        self.fingerprint = None
        self.diagnostics = {}

    def log_odds(self, features):
        """Return finite S/B log odds; balanced BCE needs no prior correction."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != len(FEATURES):
            raise ValueError(f"Preselection expects an (N, {len(FEATURES)}) feature matrix.")
        result = np.empty(len(features), dtype=np.float64)
        torch = _torch()
        with torch.no_grad():
            for start in range(0, len(features), self.config["prediction_batch_size"]):
                stop = min(start + self.config["prediction_batch_size"], len(features))
                chunk = features[start:stop]
                if not np.isfinite(chunk).all():
                    raise ValueError("Preselection input contains NaN or infinity.")
                scaled = np.asarray((chunk - self.offset) * self.scale - 1.5, dtype=np.float32)
                if not np.isfinite(scaled).all():
                    raise FloatingPointError("Preselection scaling produced non-finite values.")
                result[start:stop] = self.model(
                    torch.from_numpy(scaled).to(self.device)
                ).detach().cpu().numpy().reshape(-1)
        if not np.isfinite(result).all():
            raise FloatingPointError("Preselection classifier produced non-finite log odds.")
        return result

    def __call__(self, features):
        if self.ratio_cut is None:
            raise RuntimeError("Calibrate a preselection cut before selecting events.")
        # Clipping is used consistently for the histogram and its resulting
        # cut; a cut at the first histogram edge therefore accepts every event.
        return np.clip(self.log_odds(features), -20.0, 20.0) >= np.log(self.ratio_cut)


def _stream(component, n, seed, role, batch_size):
    rng = _rng(seed, role, component)
    report_every = max(batch_size, ((n + 4) // 5 // batch_size) * batch_size)
    next_report = report_every
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        yield simulate_reconstructed(component, stop - start, rng)
        if stop >= next_report or stop == n:
            print(f"  {role}, {component}: {stop:,}/{n:,} events.", flush=True)
            next_report = stop + report_every


def _training_bank(path, component, n, seed, role):
    bank = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, len(FEATURES)))
    start = 0
    for chunk in _stream(component, n, seed, role, _TRAIN_GENERATION_BATCH_SIZE):
        bank[start:start + len(chunk)] = chunk
        start += len(chunk)
    bank.flush()
    return bank


def _choose_cut(signal_histogram, background_histogram, inclusive_yields, target):
    """Choose the loosest supported histogram cut reaching the nominal B/S target."""
    signal = np.asarray(signal_histogram, dtype=np.int64)
    background = np.asarray(background_histogram, dtype=np.int64)
    if signal.shape != (4000,) or background.shape != signal.shape:
        raise ValueError("Calibration histograms must each contain 4,000 bins.")
    if np.any(signal < 0) or np.any(background < 0) or min(signal.sum(), background.sum()) <= 0:
        raise ValueError("Calibration histograms must contain non-negative counts and positive totals.")
    signal_accepts = np.cumsum(signal[::-1])[::-1]
    background_accepts = np.cumsum(background[::-1])[::-1]
    signal_yields = inclusive_yields[0] * signal_accepts / signal.sum()
    background_yields = inclusive_yields[1] * background_accepts / background.sum()
    ratios = np.divide(background_yields, signal_yields, out=np.full(4000, np.inf), where=signal_yields > 0)
    supported = ((signal_accepts >= _MINIMUM_CALIBRATION_ACCEPTS)
                 & (background_accepts >= _MINIMUM_CALIBRATION_ACCEPTS))
    valid = np.flatnonzero(supported & (ratios <= target))
    if not len(valid):
        raise RuntimeError(
            f"Preselection cannot reach B/S <= {target:g} with at least "
            f"{_MINIMUM_CALIBRATION_ACCEPTS} accepted calibration events in each class. "
            "Increase calibration statistics or improve the classifier; choose a looser "
            "target explicitly only if that is the intended analysis. No sparse-tail cut was saved."
        )
    best = int(valid[0])
    log_cut = float(_LOG_ODDS_EDGES[best])
    return float(np.exp(log_cut)), {
        "log_ratio_cut": log_cut,
        "histogram_signal_yield": float(signal_yields[best]),
        "histogram_background_yield": float(background_yields[best]),
        "histogram_background_to_signal": float(ratios[best]),
        "calibration_accepted": [int(signal_accepts[best]), int(background_accepts[best])],
        "calibration_totals": [int(signal.sum()), int(background.sum())],
        "minimum_calibration_accepts": _MINIMUM_CALIBRATION_ACCEPTS,
    }


def _load_classifier(directory, contract, cfg, device):
    manifest = _verified_manifest(directory, contract)
    torch = _torch()
    try:
        with np.load(directory / "weights.npz", allow_pickle=False) as payload:
            weights = {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        with np.load(directory / "scaler.npz", allow_pickle=False) as payload:
            offset, scale = payload["offset"].copy(), payload["scale"].copy()
        if (offset.shape != (len(FEATURES),) or scale.shape != offset.shape
                or not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Invalid MinMax scaler")
        if any(not bool(torch.isfinite(value).all().item()) for value in weights.values()):
            raise ValueError("Non-finite classifier weights")
        model = _network(cfg).to(device)
        model.load_state_dict(weights, strict=True)
        model.eval()
        history = json.loads((directory / "history.json").read_text())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise CacheError(f"Invalid preselection classifier cache {directory}: {error}") from error
    return Selection(model, (offset, scale), cfg, device, history), manifest


def train_or_load_selection(
    root, *, seed=12345, training_config=None,
    n_train_per_class=500_000, n_validation_per_class=100_000,
    n_calibration_per_class=5_000_000, n_yield_per_class=5_000_000,
    inclusive_yields=(1100.0, 1_000_000.0), target_background_to_signal=250.0,
    batch_size=100_000,
):
    """Train/reuse a standalone selection, calibrate its cut, and freeze yields.

    The default classifier is one 3x256 SiLU network trained with balanced BCE,
    NAdam (constant 1e-3), no regularization, and train-only MinMax[-1.5, 1.5].
    Training lasts up to 30 epochs with patience 8 and best-validation weights.
    This discriminating cut is separate from the downstream NRE estimators.

    The cut targets the supplied inclusive yields times calibration acceptance.
    Final selected yields are estimated once using fresh independent simulation;
    they have binomial Monte Carlo errors, so their B/S can fluctuate around the
    calibration target. No cut is retuned using these final yield events.

    Classifier generation uses fixed-size chunks; ``batch_size`` controls only
    cut-calibration/yield streams and is recorded in their separate contracts.
    Completed cache contents are hash-checked and never silently overwritten.
    """
    counts = {name: _positive_integer(value, name) for name, value in {
        "n_train_per_class": n_train_per_class, "n_validation_per_class": n_validation_per_class,
        "n_calibration_per_class": n_calibration_per_class, "n_yield_per_class": n_yield_per_class,
        "batch_size": batch_size,
    }.items()}
    n_train_per_class, n_validation_per_class = counts["n_train_per_class"], counts["n_validation_per_class"]
    n_calibration_per_class, n_yield_per_class = counts["n_calibration_per_class"], counts["n_yield_per_class"]
    batch_size = counts["batch_size"]
    _rng(seed, "preselection-validation", "seed-check")
    seed = int(seed)
    inclusive_yields = np.asarray(inclusive_yields, dtype=np.float64)
    if inclusive_yields.shape != (2,) or not np.isfinite(inclusive_yields).all() or np.any(inclusive_yields <= 0):
        raise ValueError("inclusive_yields must contain positive finite signal/background yields.")
    target = float(target_background_to_signal)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_background_to_signal must be positive and finite.")
    cfg = _training_config({**DEFAULT_SELECTION_TRAINING_CONFIG, **(training_config or {})})
    if cfg["ensemble_size"] != 1:
        raise ValueError("Preselection uses one classifier; ensemble_size must be 1.")
    device = _device(cfg)
    print(f"Preselection training/inference device: {device}", flush=True)
    training_contract = {
        "version": SELECTION_VERSION, "simulator": _simulator_contract(), "seed": seed,
        "training": {k: v for k, v in cfg.items() if k not in ("device", "prediction_batch_size")},
        "n_train_per_class": n_train_per_class, "n_validation_per_class": n_validation_per_class,
        "generation_batch_size": _TRAIN_GENERATION_BATCH_SIZE,
        "training_role": "preselection-training", "validation_role": "preselection-validation",
        "scaling": "train-only MinMax[-1.5,1.5]", "optimizer": "NAdam", "objective": "balanced BCE S/B",
    }
    base = Path(root) / "preselection"
    directory = base / "classifier" / _fingerprint(training_contract)
    if not directory.exists():
        with _new_artifact(directory) as staging:
            arrays = {}
            for partition, n in (("training", n_train_per_class), ("validation", n_validation_per_class)):
                for component in ("signal", "background"):
                    name = f"{partition}_{component}"
                    arrays[name] = _training_bank(staging / (name + ".npy"), component, n, seed,
                                                  "preselection-" + partition)
            minima = np.minimum(arrays["training_signal"].min(axis=0),
                                arrays["training_background"].min(axis=0)).astype(np.float64)
            maxima = np.maximum(arrays["training_signal"].max(axis=0),
                                arrays["training_background"].max(axis=0)).astype(np.float64)
            scale = 3.0 / np.where(maxima > minima, maxima - minima, 1.0)
            print("Training the independent signal/background preselection classifier.", flush=True)
            model, history = _train_member(
                arrays["training_signal"], arrays["training_background"],
                arrays["validation_signal"], arrays["validation_background"],
                cfg, seed, device, (minima, scale),
            )
            np.savez(staging / "weights.npz", **{
                name: value.detach().cpu().numpy() for name, value in model.state_dict().items()
            })
            np.savez(staging / "scaler.npz", offset=minima, scale=scale)
            (staging / "history.json").write_text(_json(history))
            del arrays
            for path in staging.glob("*.npy"):
                path.unlink()
            _write_manifest(staging, training_contract, ["weights.npz", "scaler.npz", "history.json"])
    else:
        print("Reusing the completed preselection classifier.", flush=True)
    selection, model_manifest = _load_classifier(directory, training_contract, cfg, device)
    model_fingerprint = _fingerprint({"contract": training_contract, "files": model_manifest["files"]})
    calibration_contract = {
        "version": SELECTION_VERSION, "classifier": model_fingerprint,
        "seed": seed, "role": "preselection-cut-calibration", "n_per_class": n_calibration_per_class,
        "batch_size": batch_size, "inclusive_yields": inclusive_yields.tolist(), "target": target,
        "log_odds_range": [-20.0, 20.0], "histogram_bins": 4000,
        "minimum_accepted_per_class": _MINIMUM_CALIBRATION_ACCEPTS,
    }
    calibration_directory = base / "calibration" / _fingerprint(calibration_contract)
    if not calibration_directory.exists():
        with _new_artifact(calibration_directory) as staging:
            histograms = {}
            for component in ("signal", "background"):
                histogram = np.zeros(4000, dtype=np.int64)
                for chunk in _stream(component, n_calibration_per_class, seed,
                                     "preselection-cut-calibration", batch_size):
                    histogram += np.histogram(np.clip(selection.log_odds(chunk), -20.0, 20.0),
                                              bins=_LOG_ODDS_EDGES)[0]
                histograms[component] = histogram
            ratio_cut, diagnostics = _choose_cut(histograms["signal"], histograms["background"],
                                                  inclusive_yields, target)
            np.savez(staging / "histograms.npz", **histograms, log_odds_edges=_LOG_ODDS_EDGES)
            (staging / "cut.json").write_text(_json({"ratio_cut": ratio_cut, "diagnostics": diagnostics}))
            _write_manifest(staging, calibration_contract, ["histograms.npz", "cut.json"])
    else:
        print("Reusing the completed preselection cut calibration.", flush=True)
    calibration_manifest = _verified_manifest(calibration_directory, calibration_contract)
    cut_payload = json.loads((calibration_directory / "cut.json").read_text())
    selection.ratio_cut = float(cut_payload["ratio_cut"])
    if not np.isfinite(selection.ratio_cut) or selection.ratio_cut <= 0:
        raise CacheError(f"Invalid calibrated preselection cut in {calibration_directory}.")
    calibration_fingerprint = _fingerprint({"contract": calibration_contract, "files": calibration_manifest["files"]})
    yield_contract = {
        "version": SELECTION_VERSION, "calibration": calibration_fingerprint,
        "seed": seed, "role": "preselection-yield-estimation", "n_per_class": n_yield_per_class,
        "batch_size": batch_size, "inclusive_yields": inclusive_yields.tolist(),
    }
    yield_directory = base / "yields" / _fingerprint(yield_contract)
    if not yield_directory.exists():
        with _new_artifact(yield_directory) as staging:
            accepted = []
            for component in ("signal", "background"):
                count = sum(int(selection(chunk).sum()) for chunk in _stream(
                    component, n_yield_per_class, seed, "preselection-yield-estimation", batch_size))
                if count == 0:
                    raise RuntimeError(
                        f"No {component} event passed preselection in the independent yield bank. "
                        "Increase n_yield_per_class; no zero-yield cache was saved."
                    )
                accepted.append(count)
            acceptance = np.asarray(accepted, dtype=np.float64) / n_yield_per_class
            errors = inclusive_yields * np.sqrt(acceptance * (1.0 - acceptance) / n_yield_per_class)
            yields = inclusive_yields * acceptance
            payload = {
                "yields": yields.tolist(), "yield_mc_errors": errors.tolist(),
                "yield_accepted": accepted, "yield_generated_per_class": n_yield_per_class,
                "acceptance": acceptance.tolist(), "background_to_signal": float(yields[1] / yields[0]),
                "target_background_to_signal": target,
            }
            (staging / "yields.json").write_text(_json(payload))
            _write_manifest(staging, yield_contract, ["yields.json"])
    else:
        print("Reusing the independent selected-yield estimate.", flush=True)
    yield_manifest = _verified_manifest(yield_directory, yield_contract)
    yield_payload = json.loads((yield_directory / "yields.json").read_text())
    selection.yields = np.asarray(yield_payload["yields"], dtype=np.float64)
    if (selection.yields.shape != (2,) or not np.isfinite(selection.yields).all()
            or np.any(selection.yields <= 0)):
        raise CacheError(f"Invalid selected yields in {yield_directory}.")
    selection.fingerprint = _fingerprint({"contract": yield_contract, "files": yield_manifest["files"]})
    selection.diagnostics = {
        **cut_payload["diagnostics"], **yield_payload, "classifier_fingerprint": model_fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
        "classifier_directory": str(directory), "calibration_directory": str(calibration_directory),
        "yield_directory": str(yield_directory),
    }
    errors = yield_payload["yield_mc_errors"]
    print(f"Frozen preselection odds cut: {selection.ratio_cut:.6g}; "
          f"calibration B/S={cut_payload['diagnostics']['histogram_background_to_signal']:.4g}.", flush=True)
    print(f"Independent selected yields: S={selection.yields[0]:.6g} +/- {errors[0]:.3g}, "
          f"B={selection.yields[1]:.6g} +/- {errors[1]:.3g} (binomial MC errors); "
          f"B/S={yield_payload['background_to_signal']:.4g}.", flush=True)
    return selection
