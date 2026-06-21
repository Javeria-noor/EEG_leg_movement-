"""
train.py — Training loop + 5-fold cross-validation for all 4 classifiers.

Usage:
    python src/train.py --subjects 10 --model all --output results/
    python src/train.py --subjects 10 --model eegnet --epochs 80
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from features import CSPBandFeatures, band_power_features
from models import get_sklearn_models, EEGNet, TORCH_AVAILABLE
from utils import RANDOM_SEED, get_logger, save_json, set_seed

logger = get_logger(__name__)


def cross_validate_sklearn(X_feats: np.ndarray, y: np.ndarray, model, n_splits: int = 5):
    """Run stratified K-fold CV for a single sklearn-compatible model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    accs, f1s, aucs = [], [], []
    last_cm = None

    for train_idx, test_idx in skf.split(X_feats, y):
        model.fit(X_feats[train_idx], y[train_idx])
        preds = model.predict(X_feats[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro"))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_feats[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))
        last_cm = confusion_matrix(y[test_idx], preds)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "auc_roc_mean": float(np.mean(aucs)) if aucs else None,
        "confusion_matrix": last_cm.tolist() if last_cm is not None else None,
    }


def train_eegnet(X: np.ndarray, y: np.ndarray, n_epochs: int = 80, lr: float = 1e-3):
    """Train EEGNet on raw epoch tensors with a simple train/val split per fold."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for EEGNet training: pip install torch")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(RANDOM_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    accs, f1s, aucs = [], [], []
    history = {"train_loss": [], "val_loss": []}
    last_cm = None

    n_channels, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        Xtr = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1)
        ytr = torch.tensor(y[train_idx], dtype=torch.long)
        Xte = torch.tensor(X[test_idx], dtype=torch.float32).unsqueeze(1)
        yte = torch.tensor(y[test_idx], dtype=torch.long)

        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)

            model.eval()
            with torch.no_grad():
                val_out = model(Xte)
                val_loss = crit(val_out, yte).item()

            if fold == 0:
                history["train_loss"].append(epoch_loss)
                history["val_loss"].append(val_loss)

        model.eval()
        with torch.no_grad():
            logits = model(Xte)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).numpy()

        accs.append(accuracy_score(yte.numpy(), preds))
        f1s.append(f1_score(yte.numpy(), preds, average="macro"))
        if n_classes == 2:
            aucs.append(roc_auc_score(yte.numpy(), probs[:, 1].numpy()))
        last_cm = confusion_matrix(yte.numpy(), preds)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "auc_roc_mean": float(np.mean(aucs)) if aucs else None,
        "confusion_matrix": last_cm.tolist() if last_cm is not None else None,
        "history": history,
    }


def run_all(X_raw: np.ndarray, y: np.ndarray, models_to_run: str = "all",
            n_epochs: int = 80) -> list:
    """Run the full model comparison and return a list of result dicts."""
    set_seed(RANDOM_SEED)
    results = []

    if models_to_run in ("all", "classical"):
        logger.info("Extracting CSP band-power features...")
        csp = CSPBandFeatures()
        X_feats = csp.fit_transform(X_raw, y)

        for name, model in get_sklearn_models().items():
            logger.info("Cross-validating %s...", name)
            metrics = cross_validate_sklearn(X_feats, y, model)
            metrics["model"] = name
            results.append(metrics)
            logger.info("%s: acc=%.3f +/- %.3f", name,
                        metrics["accuracy_mean"], metrics["accuracy_std"])

    if models_to_run in ("all", "eegnet"):
        logger.info("Training EEGNet CNN...")
        metrics = train_eegnet(X_raw, y, n_epochs=n_epochs)
        metrics["model"] = "EEGNet CNN"
        results.append(metrics)
        logger.info("EEGNet CNN: acc=%.3f +/- %.3f",
                    metrics["accuracy_mean"], metrics["accuracy_std"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BCI classifiers")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--model", type=str, default="all",
                         choices=["all", "classical", "eegnet"])
    parser.add_argument("--epochs", type=int, default=80,
                         help="Training epochs for EEGNet")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--data", type=str, default="data/",
                         help="Path to preprocessed epoch arrays (X.npy, y.npy)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    X_path, y_path = data_dir / "X.npy", data_dir / "y.npy"
    if not X_path.exists() or not y_path.exists():
        logger.error(
            "Preprocessed arrays not found at %s. Run preprocessing first "
            "(see notebooks/02_preprocessing.ipynb).", data_dir
        )
        return

    X, y = np.load(X_path), np.load(y_path)
    logger.info("Loaded data: X=%s, y=%s", X.shape, y.shape)

    results = run_all(X, y, models_to_run=args.model, n_epochs=args.epochs)

    out_path = Path(args.output) / "metrics.json"
    save_json(results, out_path)
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
