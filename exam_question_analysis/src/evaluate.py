"""
evaluate.py
-----------
Metrics, confusion matrix, and comparison reporting for all trained models.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    r2_score,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Easy", "Medium", "Hard"]

def evaluate_model(clf, X_test, y_test, model_name: str, save_dir: str) -> dict:
    """
    Evaluate a trained classifier and save confusion matrix.

    Returns dict with accuracy, f1_weighted, r2 (supplementary), report_str.
    """
    y_pred = clf.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_w       = f1_score(y_test, y_pred, average="weighted")
    r2         = r2_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    cm         = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Model: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 (wtd)  : {f1_w:.4f}")
    print(f"  R² (suppl): {r2:.4f}  ← regression metric, for reference only")
    print(f"\nClassification Report:")
    print(report_str)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    img_path  = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(img_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {img_path}")

    return {
        "accuracy":    acc,
        "f1_weighted": f1_w,
        "r2":          r2,
        "report":      report_str,
        "cm":          cm.tolist(),
        "cm_img_path": img_path,
    }

def print_comparison_table(results: dict):
    """Print a formatted comparison table for all models."""
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 55)
    header = f"{'Model':<28} {'Accuracy':>10} {'F1 (wtd)':>12} {'R² (suppl)':>12}"
    print(header)
    print("-" * 55)
    for name, m in sorted(results.items(), key=lambda x: -x[1]["f1_weighted"]):
        print(
            f"{name:<28} {m['accuracy']:>9.4f}  {m['f1_weighted']:>10.4f}  {m['r2']:>10.4f}"
        )
    print("=" * 55)
