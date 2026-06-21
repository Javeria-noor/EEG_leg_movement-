"""
utils.py — Shared configuration, logging, and helper utilities.
"""
import json
import logging
import random
from pathlib import Path

import numpy as np

# ── Global config ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

SFREQ = 160                      # Hz, BCI2000 native sampling rate
CHANNELS = [
    "Cz", "FCz", "CPz", "C3", "C4", "Pz", "Fz",
    "P3", "P4", "O1", "O2", "F3", "F4",
]
MIDLINE_CHANNELS = {"Cz", "FCz", "CPz", "Pz", "Fz"}

# Runs in the BCI2000 / OpenNeuro ds004362 protocol that contain
# imagined "both feet" trials (T2 in runs 6/10/14) used for the
# lower-limb decoding task in this project.
LEG_IMAGERY_RUNS = [6, 10, 14]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),   # primary ERD band for foot motor imagery
    "beta": (13, 30),   # secondary ERD band
    "gamma": (30, 40),
}

EPOCH_TMIN, EPOCH_TMAX = -0.5, 2.5   # seconds relative to cue onset
BASELINE = (-0.5, 0.0)
PEAK_TO_PEAK_REJECT_UV = 100e-6      # volts (MNE convention)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed numpy / random / torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "eeg_leg_bci") -> logging.Logger:
    """Return a configured module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
