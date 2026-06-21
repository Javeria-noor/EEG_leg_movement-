# Methodology

Detailed write-up of the preprocessing, feature extraction, and modelling
choices behind the EEG Lower Limb Motor Imagery BCI pipeline.

## 1. Dataset

[OpenNeuro `ds004362`](https://openneuro.org/datasets/ds004362) (PhysioNet /
BCI2000 Motor Movement/Imagery dataset), CC0 licensed.

- 109 healthy subjects, 64-channel EEG, 160 Hz sampling rate, EDF+/BIDS format
- 14 runs per subject (2 baseline + 12 task runs)
- This project uses **runs 6, 10, 14** — the imagined "both feet" condition,
  used here as the basis for the lower-limb (leg) decoding task and for the
  4-way wheelchair command set (`left` / `right` / `both` / `stop`) explored
  in the dashboard.

### Event codes

| Code | Meaning (leg-imagery runs) |
|------|------------------------------|
| `T0` | Rest |
| `T1` | Left fist / left foot imagery (run-dependent) |
| `T2` | Right fist / both feet imagery (run-dependent) |

## 2. Why midline electrodes?

The leg representation in the homunculus sits at the **vertex / midline**
of the motor cortex (around `Cz`, `FCz`, `CPz`), unlike hand motor imagery
which is lateralised to `C3`/`C4`. Alpha (8–13 Hz) and beta (13–30 Hz)
event-related desynchronisation (ERD) at these midline channels is the
primary neural signature this pipeline decodes.

## 3. Preprocessing pipeline

```
Raw EEG (64ch, 160 Hz)
    │
    ▼
Bandpass filter: 1–40 Hz (5th-order Butterworth, zero-phase)
    │
    ▼
Notch filter: 50 Hz (power-line interference)
    │
    ▼
ICA (FastICA, 20 components) → remove eye/muscle artifacts
    │
    ▼
Epoching: T1/T2 events → [-0.5, 2.5 s] windows relative to cue onset
    │
    ▼
Baseline correction: [-0.5, 0] s
    │
    ▼
Rejection: epochs with peak-to-peak amplitude > 100 µV removed
```

Implemented in `src/preprocessing.py`.

## 4. Feature extraction

Two complementary feature sets are computed (`src/features.py`):

**Option A — CSP band-power (primary feature set)**
- Common Spatial Patterns (6 components) fit independently per frequency
  band (δ, θ, α, β, γ)
- Log-variance of the CSP-filtered signal as the feature value
- Final vector: 5 bands × 6 components = 30 features (used by the classical
  models; the dashboard's CSP feature-importance panel mirrors this layout)

**Option B — Welch PSD band power**
- `scipy.signal.welch` PSD, `nperseg=160` (1 s windows, 50% overlap)
- Mean power per band per channel → 5 bands × n_channels features
- Reduced to 50 dimensions via PCA before feeding to classical models

## 5. Models

| Model | Features | Notes |
|---|---|---|
| Logistic Regression (L2) | CSP band power | Linear baseline |
| SVM (RBF kernel) | CSP band power | Best classical performer in most runs |
| Random Forest (300 trees) | CSP + PSD | Gives feature-importance ranking |
| **EEGNet (CNN)** | Raw epochs | Compact architecture, end-to-end learned spatial+temporal filters |

### EEGNet architecture

```
Input  (1, n_channels, n_times)
  → Temporal Conv2D (1, 1, 64) + BatchNorm
  → Depthwise Conv2D + BatchNorm + ELU + AvgPool + Dropout(0.5)
  → Separable Conv2D + BatchNorm + ELU + AvgPool + Dropout(0.5)
  → Flatten → Dense(n_classes) → Softmax
```

Reference: Lawhern et al., *"EEGNet: a compact convolutional neural network
for EEG-based brain–computer interfaces"*, J. Neural Eng., 2018.

## 6. Evaluation

- 5-fold stratified cross-validation for every model
- Metrics: accuracy, macro F1, AUC-ROC, confusion matrix
- Results are written to `results/metrics.json` and consumed directly by
  `app.py` (the Streamlit dashboard) and `src/evaluate.py` (static figure
  generation)

## 7. Wheelchair command simulation

The dashboard's "Wheelchair Sim" tab maps the 4-class decode
(`left` / `right` / `both` / `stop`) onto a simple grid-navigation demo.
A confidence threshold gates which trials are "executed" vs "rejected",
illustrating how a deployed system would use majority voting across
consecutive epochs to suppress spurious single-epoch misclassifications
before sending a command to actual hardware.

## 8. Reproducibility

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

set via `src/utils.set_seed()`, called at the start of every training/notebook
entry point.
