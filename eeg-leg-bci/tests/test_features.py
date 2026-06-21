"""
test_features.py — Unit tests for band power, CSP, and PSD feature extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import band_power_features, bandpass_envelope, psd
from utils import BANDS, SFREQ


def _synthetic_signal(n_seconds=3, sfreq=SFREQ, freq=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_seconds, int(sfreq * n_seconds))
    return t, np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(len(t))


def test_bandpass_envelope_shape():
    _, sig = _synthetic_signal()
    env = bandpass_envelope(sig, 8, 13)
    assert env.shape == sig.shape
    assert np.all(env >= 0)  # envelope is non-negative


def test_bandpass_envelope_isolates_alpha():
    """A pure 10 Hz signal filtered to alpha (8-13Hz) should retain most of its power;
    filtered to gamma (30-40Hz) should have near-zero envelope."""
    _, sig = _synthetic_signal(freq=10)
    alpha_env = bandpass_envelope(sig, 8, 13)
    gamma_env = bandpass_envelope(sig, 30, 40)
    assert alpha_env.mean() > gamma_env.mean() * 5


def test_band_power_features_shape():
    n_epochs, n_channels = 4, 13
    X = np.random.default_rng(1).standard_normal((n_epochs, n_channels, SFREQ * 3))
    feats = band_power_features(X)
    assert feats.shape == (n_epochs, len(BANDS) * n_channels)
    assert np.all(np.isfinite(feats))


def test_band_power_nonnegative():
    X = np.random.default_rng(2).standard_normal((2, 3, SFREQ * 2))
    feats = band_power_features(X)
    assert np.all(feats >= 0)


def test_psd_truncated_to_fmax():
    _, sig = _synthetic_signal()
    f, p = psd(sig, fmax=50.0)
    assert f.max() <= 50.0
    assert f.shape == p.shape


def test_psd_peak_near_signal_frequency():
    _, sig = _synthetic_signal(freq=10)
    f, p = psd(sig)
    peak_freq = f[np.argmax(p)]
    assert peak_freq == pytest.approx(10, abs=2)
