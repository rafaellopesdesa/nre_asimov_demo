"""Checks for independent preselection calibration and immutable reuse."""

import json
from pathlib import Path

import numpy as np
import pytest

import utils_preselection as selection_module
from utils_nre import CacheError, _rng


def test_cut_is_loosest_supported_cut():
    signal = np.zeros(4000, dtype=np.int64)
    background = np.zeros(4000, dtype=np.int64)
    signal[2000], signal[2100] = 100, 100
    background[2000], background[2100] = 1000, 100
    cut, diagnostics = selection_module._choose_cut(signal, background, np.array([1.0, 10.0]), 2.0)
    assert cut == np.exp(selection_module._LOG_ODDS_EDGES[2001])
    assert diagnostics["calibration_accepted"] == [100, 100]
    assert diagnostics["histogram_background_to_signal"] == pytest.approx(20 / 11)


def test_cut_refuses_sparse_background_tail():
    signal = np.zeros(4000, dtype=np.int64)
    background = np.zeros(4000, dtype=np.int64)
    signal[2000], signal[2100] = 100, 100
    background[2000], background[2100] = 1099, 1
    with pytest.raises(RuntimeError, match="No sparse-tail cut"):
        selection_module._choose_cut(signal, background, np.array([1.0, 10.0]), 2.0)


def test_random_roles_are_reproducible_and_independent():
    roles = ("training", "validation", "cut-calibration", "yield-estimation")
    streams = [_rng(12345, "preselection-" + role, "signal").normal(size=32) for role in roles]
    for i, role in enumerate(roles):
        np.testing.assert_array_equal(streams[i], _rng(12345, "preselection-" + role, "signal").normal(size=32))
        for j in range(i):
            assert not np.array_equal(streams[i], streams[j])


@pytest.fixture
def tiny_selection_options():
    return dict(
        seed=1729, n_train_per_class=64, n_validation_per_class=32,
        n_calibration_per_class=200, n_yield_per_class=200, batch_size=100,
        target_background_to_signal=2000.0,
        training_config={"hidden_layers": 1, "hidden_features": 4,
                         "epochs": 1, "batch_size": 32, "device": "cpu"},
    )


def test_yield_changes_reuse_classifier_and_calibration(tmp_path, monkeypatch, tiny_selection_options):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    initial = selection_module.train_or_load_selection(tmp_path, **tiny_selection_options)
    classifier = Path(initial.diagnostics["classifier_directory"])
    calibration = Path(initial.diagnostics["calibration_directory"])
    original_weights = (classifier / "weights.npz").read_bytes()
    original_cut = (calibration / "cut.json").read_bytes()

    def cannot_retrain(*args, **kwargs):
        raise AssertionError("A downstream bank change must not retrain the selection classifier")

    monkeypatch.setattr(selection_module, "_train_member", cannot_retrain)
    reused = selection_module.train_or_load_selection(tmp_path, **tiny_selection_options)
    assert reused.fingerprint == initial.fingerprint
    larger = selection_module.train_or_load_selection(
        tmp_path, **{**tiny_selection_options, "n_yield_per_class": 300})
    assert larger.diagnostics["classifier_fingerprint"] == initial.diagnostics["classifier_fingerprint"]
    assert larger.diagnostics["calibration_fingerprint"] == initial.diagnostics["calibration_fingerprint"]
    assert larger.fingerprint != initial.fingerprint
    assert (classifier / "weights.npz").read_bytes() == original_weights
    assert (calibration / "cut.json").read_bytes() == original_cut
    assert larger.diagnostics["yield_generated_per_class"] == 300
    np.testing.assert_array_equal(larger.yields, [1100.0, 1_000_000.0])
    np.testing.assert_array_equal(larger(np.empty((0, 5))), np.empty(0, dtype=bool))


def test_corrupt_classifier_is_never_retrained(tmp_path, monkeypatch, tiny_selection_options):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    initial = selection_module.train_or_load_selection(tmp_path, **tiny_selection_options)
    path = Path(initial.diagnostics["classifier_directory"]) / "weights.npz"
    corrupted = path.read_bytes() + b"corruption"
    path.write_bytes(corrupted)

    def cannot_retrain(*args, **kwargs):
        raise AssertionError("Corruption must raise instead of retraining")

    monkeypatch.setattr(selection_module, "_train_member", cannot_retrain)
    with pytest.raises(CacheError, match="hash mismatch"):
        selection_module.train_or_load_selection(tmp_path, **tiny_selection_options)
    assert path.read_bytes() == corrupted
