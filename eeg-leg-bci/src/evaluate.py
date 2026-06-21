"""
evaluate.py — Metrics summarisation, confusion matrices, and result plots.

Reads results/metrics.json (produced by train.py) and generates the
figures referenced in the README / dashboard: scalp topomaps, CSP
spatial filters, confusion matrix heatmaps, and learning curves.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import RESULTS_DIR, get_logger, load_json

logger = get_logger(__name__)


def summarize(results: list) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Model':<22}{'Accuracy':>12}{'F1 (macro)':>14}{'AUC-ROC':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        acc = f"{r['accuracy_mean']*100:.1f}% ± {r['accuracy_std']*100:.1f}%"
        f1 = f"{r['f1_macro_mean']:.3f}"
        auc = f"{r['auc_roc_mean']:.3f}" if r.get("auc_roc_mean") is not None else "—"
        print(f"{r['model']:<22}{acc:>12}{f1:>14}{auc:>12}")


def plot_confusion_matrices(results: list, out_dir: Path,
                              class_names=("Left", "Right")) -> None:
    """Save one confusion-matrix heatmap per model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        cm = r.get("confusion_matrix")
        if cm is None:
            continue
        cm = np.array(cm, dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(r["model"])
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()

        fname = out_dir / f"confusion_{r['model'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", fname)


def plot_accuracy_comparison(results: list, out_dir: Path) -> None:
    """Bar chart comparing accuracy / F1 / AUC across models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results]
    acc = [r["accuracy_mean"] for r in results]
    f1 = [r["f1_macro_mean"] for r in results]
    auc = [r.get("auc_roc_mean") or 0 for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, acc, width, label="Accuracy")
    ax.bar(x, f1, width, label="F1 (macro)")
    ax.bar(x + width, auc, width, label="AUC-ROC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Comparison")
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "model_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", fname)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise results")
    parser.add_argument("--metrics", type=str, default=str(RESULTS_DIR / "metrics.json"))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args()

    results = load_json(args.metrics)
    summarize(results)

    out_dir = Path(args.output)
    plot_confusion_matrices(results, out_dir)
    plot_accuracy_comparison(results, out_dir)


if __name__ == "__main__":
    main()
