"""
SoundSentinel — Evaluation Script
===================================
Loads a trained SoundSentinel model, evaluates it on the ESC-50 dataset,
and generates diagnostic visualizations saved to training/outputs/.

Usage
-----
    python evaluate.py --data-dir data/ESC-50 --model-path models/sound_classifier.pkl

Outputs
-------
    training/outputs/confusion_matrix.png   — Normalized confusion matrix heatmap
    training/outputs/per_class_accuracy.png — Per-class accuracy bar chart
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import shared feature extractor from train.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from train import extract_features, SAMPLE_RATE, DURATION


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    Render a normalized confusion matrix heatmap and save to disk.

    Parameters
    ----------
    cm          : Raw (unnormalized) confusion matrix from sklearn
    class_names : Ordered list of class label strings
    output_path : Where to write the PNG file
    """
    # Normalize row-wise so each row sums to 1.0
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n_classes = len(class_names)
    fig_size = max(18, n_classes * 0.45)  # Scale figure with class count

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor="#2a2a4e",
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": 6},
        vmin=0.0,
        vmax=1.0,
    )

    # Style the axes
    ax.set_title(
        "SoundSentinel — Normalized Confusion Matrix (ESC-50)",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, color="#a0a0c0", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, color="#a0a0c0", labelpad=10)
    ax.tick_params(colors="#a0a0c0", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Proportion", color="#a0a0c0", fontsize=10)
    cbar.ax.tick_params(colors="#a0a0c0")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {output_path}")


def plot_per_class_accuracy(
    per_class_acc: dict[str, float],
    output_path: Path,
) -> None:
    """
    Render a horizontal bar chart of per-class accuracy and save to disk.

    Parameters
    ----------
    per_class_acc : Mapping of class_name -> accuracy (0.0–1.0)
    output_path   : Where to write the PNG file
    """
    # Sort descending by accuracy
    sorted_items = sorted(per_class_acc.items(), key=lambda kv: kv[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, max(8, n * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Colour bars by accuracy tier
    colors = [
        "#00d4aa" if acc >= 0.95
        else "#4fc3f7" if acc >= 0.85
        else "#ff7043"
        for acc in accuracies
    ]

    bars = ax.barh(labels, accuracies, color=colors, edgecolor="#2a2a4e", linewidth=0.5)

    # Annotate each bar with its value
    for bar, acc in zip(bars, accuracies):
        ax.text(
            min(acc + 0.005, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            ha="left",
            color="white",
            fontsize=7.5,
        )

    # Reference line at 0.90
    ax.axvline(0.90, color="#ff9800", linestyle="--", linewidth=1.0, alpha=0.7, label="90% threshold")

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy", fontsize=12, color="#a0a0c0", labelpad=10)
    ax.set_title(
        "SoundSentinel — Per-Class Accuracy (ESC-50 Test Set)",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15,
    )
    ax.tick_params(colors="#a0a0c0", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, facecolor="#2a2a4e", labelcolor="white")

    # Grid
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#4a4a6a")
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Per-class accuracy chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Dataset loading (reuses feature extractor from train.py)
# ---------------------------------------------------------------------------

def load_test_features(
    data_dir: Path,
    test_fold: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract features from ESC-50 clips belonging to the specified test fold.

    ESC-50 has 5 predefined folds for cross-validation. By convention, fold 5
    is used as the held-out test set.

    Returns
    -------
    X_test       : float32 array of shape [N, n_features]
    y_test       : int64 array of shape [N]
    class_names  : list of 50 class name strings
    """
    audio_dir = data_dir / "audio"
    meta_path = data_dir / "meta" / "esc50.csv"

    if not meta_path.exists():
        sys.exit(
            f"[ERROR] Metadata CSV not found: {meta_path}\n"
            "Download ESC-50: git clone https://github.com/karolpiczak/ESC-50.git training/data/ESC-50"
        )

    meta = pd.read_csv(meta_path)
    test_meta = meta[meta["fold"] == test_fold]
    print(f"[INFO] Test fold {test_fold}: {len(test_meta)} clips")

    X_list, y_list = [], []
    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Extracting test features"):
        file_path = audio_dir / row["filename"]
        if not file_path.exists():
            continue
        feat = extract_features(str(file_path))
        if feat is not None:
            X_list.append(feat)
            y_list.append(row["target"])

    X_test = np.array(X_list, dtype=np.float32)
    y_test = np.array(y_list, dtype=np.int64)

    class_map = meta.drop_duplicates("target").set_index("target")["category"].to_dict()
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    return X_test, y_test, class_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SoundSentinel model and generate diagnostic plots"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ESC-50"),
        help="Path to the ESC-50 root directory (default: data/ESC-50)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/sound_classifier.pkl"),
        help="Path to the trained model .pkl file (default: models/sound_classifier.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output charts (default: outputs)",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="ESC-50 fold to use as test set (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  SoundSentinel — Model Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if not args.model_path.exists():
        sys.exit(
            f"[ERROR] Model not found: {args.model_path}\n"
            "Run training first: python train.py --data-dir data/ESC-50"
        )

    saved = joblib.load(args.model_path)
    model = saved["model"]
    class_names = saved["class_names"]
    n_features = saved["n_features"]
    print(f"[INFO] Loaded model from {args.model_path}")
    print(f"[INFO] Classes: {len(class_names)}  |  Features: {n_features}")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    X_test, y_test, _ = load_test_features(args.data_dir, test_fold=args.test_fold)

    if len(X_test) == 0:
        sys.exit("[ERROR] No test samples found — check data directory and fold number")

    print(f"[INFO] Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------
    print("[INFO] Running predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # ------------------------------------------------------------------
    # 4. Classification report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # ------------------------------------------------------------------
    # 5. Confusion matrix plot
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    cm_path = args.output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # ------------------------------------------------------------------
    # 6. Per-class accuracy bar chart
    # ------------------------------------------------------------------
    # Compute per-class accuracy from the confusion matrix diagonal
    per_class_acc = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc[name] = correct / total if total > 0 else 0.0

    acc_path = args.output_dir / "per_class_accuracy.png"
    plot_per_class_accuracy(per_class_acc, acc_path)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    best_class = max(per_class_acc, key=per_class_acc.get)
    worst_class = min(per_class_acc, key=per_class_acc.get)

    print("\n" + "=" * 60)
    print(f"  Overall Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Best class:        {best_class} ({per_class_acc[best_class] * 100:.1f}%)")
    print(f"  Worst class:       {worst_class} ({per_class_acc[worst_class] * 100:.1f}%)")
    print(f"  Outputs saved to:  {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
