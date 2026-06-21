"""
models.py — Classical sklearn classifiers + PyTorch EEGNet definition.

Architecture (matches app.py / README):
    Input (1, n_channels, n_times)
        -> Temporal Conv2D (1, 1, 64) + BN
        -> Depthwise Conv2D + BN + ELU + AvgPool + Dropout(0.5)
        -> Separable Conv2D + BN + ELU + AvgPool + Dropout(0.5)
        -> Flatten -> Dense(n_classes) -> Softmax
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_sklearn_models(seed: int = 42) -> dict:
    """Return the 4-way classical model bank used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=seed,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1,
        ),
    }


# ── EEGNet (PyTorch) ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        """
        Compact CNN for EEG-based BCI, after Lawhern et al. 2018.

        Parameters
        ----------
        n_channels : number of EEG channels (e.g. 13 for the midline+lateral
                     leg-imagery montage, or 64 for the full cap)
        n_times    : number of time samples per epoch
        n_classes  : number of output classes (2 for left/right or
                     4 for left/right/both/stop command decoding)
        """

        def __init__(self, n_channels: int = 13, n_times: int = 481,
                     n_classes: int = 2, F1: int = 8, D: int = 2,
                     dropout: float = 0.5):
            super().__init__()
            F2 = F1 * D

            self.firstconv = nn.Sequential(
                nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(F1),
            )
            self.depthwise = nn.Sequential(
                nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
                nn.Conv2d(F2, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                flat_dim = self._forward_features(dummy).shape[1]

            self.classify = nn.Linear(flat_dim, n_classes)

        def _forward_features(self, x):
            x = self.firstconv(x)
            x = self.depthwise(x)
            x = self.separable(x)
            return x.flatten(start_dim=1)

        def forward(self, x):
            x = self._forward_features(x)
            return self.classify(x)

    TORCH_AVAILABLE = True

except ImportError:
    EEGNet = None
    TORCH_AVAILABLE = False
