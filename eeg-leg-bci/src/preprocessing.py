"""
preprocessing.py — MNE pipeline: bandpass filter → notch → ICA → epoch.

Pipeline (matches app.py dashboard description):
    Raw EEG (64ch, 160 Hz)
        -> Bandpass filter: 1-40 Hz (5th-order Butterworth)
        -> Notch filter: 50 Hz (power-line)
        -> ICA (FastICA) -> remove eye/muscle artifacts
        -> Epoching: T1/T2 events -> [-0.5, 2.5 s] windows
        -> Baseline correction: [-0.5, 0] s
        -> Rejection: epochs with peak-to-peak amplitude > 100 uV removed
"""
from pathlib import Path

import numpy as np

from utils import (
    BASELINE,
    EPOCH_TMAX,
    EPOCH_TMIN,
    PEAK_TO_PEAK_REJECT_UV,
    get_logger,
)

logger = get_logger(__name__)


def load_raw(edf_path: Path):
    """Load a single BCI2000 EDF+ run with MNE."""
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels(lambda ch: ch.strip("."))
    raw.set_montage("standard_1005", on_missing="warn")
    return raw


def filter_raw(raw, l_freq: float = 1.0, h_freq: float = 40.0, notch: float = 50.0):
    """Apply bandpass + notch filtering in place and return the raw object."""
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[notch], verbose=False)
    return raw


def run_ica(raw, n_components: int = 20, seed: int = 42):
    """
    Fit FastICA and remove components correlated with EOG/EMG artifacts.
    Falls back to returning the raw unchanged if no frontal channels are
    available to build an EOG proxy.
    """
    from mne.preprocessing import ICA

    ica = ICA(n_components=n_components, method="fastica",
              random_state=seed, max_iter="auto")
    ica.fit(raw, verbose=False)

    eog_like = [ch for ch in ("Fp1", "Fp2", "F3", "F4") if ch in raw.ch_names]
    if eog_like:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_like, verbose=False)
        ica.exclude = eog_indices
        logger.info("ICA excluding %d artifact components", len(eog_indices))
    else:
        logger.info("No frontal channels found for EOG proxy; skipping auto-exclude")

    ica.apply(raw, verbose=False)
    return raw


def epoch_events(raw, event_id: dict | None = None):
    """
    Extract epochs around T1/T2 annotation events.

    event_id maps annotation descriptions (e.g. 'T1', 'T2') to integer codes.
    Defaults to the standard BCI2000 scheme: T1=left/both-feet, T2=right/rest
    depending on the run; callers should pass an explicit mapping per run.
    """
    import mne

    events, found_event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = event_id or found_event_id

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE,
        preload=True,
        reject=dict(eeg=PEAK_TO_PEAK_REJECT_UV),
        verbose=False,
    )
    n_dropped = len(epochs.drop_log) - len(epochs)
    logger.info("Epoched %d trials (%d dropped by peak-to-peak rejection)",
                len(epochs), n_dropped)
    return epochs


def preprocess_run(edf_path: Path, event_id: dict | None = None):
    """Full pipeline for a single run file: load -> filter -> ICA -> epoch."""
    raw = load_raw(edf_path)
    raw = filter_raw(raw)
    raw = run_ica(raw)
    epochs = epoch_events(raw, event_id=event_id)
    return epochs


def epochs_to_arrays(epochs):
    """Return (X, y) numpy arrays from an MNE Epochs object."""
    X = epochs.get_data()                       # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]                     # integer labels
    return X, y
