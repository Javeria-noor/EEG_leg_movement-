"""
test_preprocessing.py — Unit tests for the MNE preprocessing pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import BASELINE, EPOCH_TMAX, EPOCH_TMIN, PEAK_TO_PEAK_REJECT_UV, SFREQ


def test_epoch_window_length():
    """Epoch window should span exactly 3 seconds (TMIN to TMAX)."""
    assert EPOCH_TMAX - EPOCH_TMIN == pytest.approx(3.0)


def test_baseline_within_epoch():
    """Baseline window must lie within the epoch window."""
    assert EPOCH_TMIN <= BASELINE[0]
    assert BASELINE[1] <= EPOCH_TMAX


def test_reject_threshold_positive():
    assert PEAK_TO_PEAK_REJECT_UV > 0


def test_sfreq_matches_bci2000():
    assert SFREQ == 160


def test_filter_raw_runs():
    """filter_raw should call MNE's filter/notch_filter without error on a stub raw."""
    from preprocessing import filter_raw

    class StubRaw:
        def __init__(self):
            self.filtered = False
            self.notched = False

        def filter(self, l_freq, h_freq, fir_design="firwin", verbose=False):
            assert l_freq == 1.0 and h_freq == 40.0
            self.filtered = True

        def notch_filter(self, freqs, verbose=False):
            assert freqs == [50]
            self.notched = True

    raw = StubRaw()
    result = filter_raw(raw)
    assert result.filtered and result.notched


def test_epochs_to_arrays_shapes():
    from preprocessing import epochs_to_arrays

    class StubEpochs:
        def get_data(self):
            return np.zeros((10, 13, 481))

        events = np.array([[i, 0, i % 2] for i in range(10)])

    X, y = epochs_to_arrays(StubEpochs())
    assert X.shape == (10, 13, 481)
    assert y.shape == (10,)
    assert set(np.unique(y)).issubset({0, 1})
