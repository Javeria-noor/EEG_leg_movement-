"""
features.py — Band power, Common Spatial Patterns (CSP), and PSD features.

Matches the two feature-extraction options used in the dashboard:
  Option A: CSP log-variance per band  -> 6 bands x 6 CSP comps = 36 features
  Option B: Welch PSD mean power per band x channel -> 5 bands x 64ch (-> PCA)
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from utils import BANDS, SFREQ, get_logger

logger = get_logger(__name__)


def bandpass_envelope(signal: np.ndarray, lo: float, hi: float,
                       sfreq: int = SFREQ, order: int = 5) -> np.ndarray:
    """Bandpass-filter a 1D signal and return its Hilbert amplitude envelope."""
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signal)
    return np.abs(hilbert(filtered))


def band_power_features(X: np.ndarray, sfreq: int = SFREQ,
                         bands: dict | None = None) -> np.ndarray:
    """
    Welch PSD band-power features.

    Parameters
    ----------
    X : array, shape (n_epochs, n_channels, n_times)

    Returns
    -------
    features : array, shape (n_epochs, n_bands * n_channels)
    """
    bands = bands or BANDS
    n_epochs, n_channels, _ = X.shape
    feats = np.zeros((n_epochs, len(bands) * n_channels))

    for e in range(n_epochs):
        col = 0
        for ch in range(n_channels):
            f, p = welch(X[e, ch], fs=sfreq, nperseg=sfreq)
            for lo, hi in bands.values():
                mask = (f >= lo) & (f <= hi)
                feats[e, col] = np.mean(p[mask]) if mask.any() else 0.0
                col += 1
    return feats


class CSPBandFeatures:
    """
    Common Spatial Patterns features computed per frequency band, then
    log-variance is taken as the final feature (the dashboard's
    "Option A" feature set: n_bands x n_csp_components).

    Wraps mne.decoding.CSP per band and concatenates the outputs.
    """

    def __init__(self, bands: dict | None = None, n_components: int = 6,
                 sfreq: int = SFREQ):
        self.bands = bands or BANDS
        self.n_components = n_components
        self.sfreq = sfreq
        self._csp_by_band = {}

    def _filter_band(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        sos = butter(5, [lo, hi], btype="band", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from mne.decoding import CSP

        all_feats = []
        for name, (lo, hi) in self.bands.items():
            Xb = self._filter_band(X, lo, hi)
            csp = CSP(n_components=self.n_components, log=True,
                      norm_trace=False)
            feats = csp.fit_transform(Xb, y)
            self._csp_by_band[name] = csp
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        all_feats = []
        for name, (lo, hi) in self.bands.items():
            csp = self._csp_by_band[name]
            Xb = self._filter_band(X, lo, hi)
            all_feats.append(csp.transform(Xb))
        return np.concatenate(all_feats, axis=1)

    def feature_names(self) -> list:
        names = []
        for band in self.bands:
            for c in range(1, self.n_components + 1):
                names.append(f"{band}-CSP{c}")
        return names


def psd(signal: np.ndarray, sfreq: int = SFREQ, fmax: float = 50.0):
    """Single-channel Welch PSD in dB, truncated to fmax (used by Signal Explorer)."""
    f, p = welch(signal, fs=sfreq, nperseg=sfreq)
    mask = f <= fmax
    return f[mask], 10 * np.log10(p[mask] + 1e-12)
