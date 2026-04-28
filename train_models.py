"""
Train and evaluate two galaxy morphology classifiers
=====================================================
Compares two classical machine-learning baselines on the synthetic
galaxy dataset, then produces figures for the case study:

    1. HOG (Histogram of Oriented Gradients) features + Random Forest
    2. Raw-pixel features + Multi-Layer Perceptron (MLP)

These provide reproducible baselines that demonstrate the
machine-learning workflow described in the paper. The full reproducible
CNN counterpart is provided in the accompanying Jupyter notebook
(see github.com/.../notebook.ipynb), which uses PyTorch and a small
ResNet-style architecture.

Outputs (figures/):
    - fig_training_curves.png
    - fig_confusion_matrix.png
    - fig_predictions.png
    - results_summary.json
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from skimage.feature import hog

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "galaxy_dataset.npz"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ["Elliptical", "Spiral", "Irregular"]
RNG_SEED = 42


def load_data():
    data = np.load(DATA_PATH)
    return data["X"], data["y"]


def extract_hog_features(images):
    """Extract HOG descriptors for each image."""
    feats = []
    for img in images:
        f = hog(img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys",
                feature_vector=True)
        feats.append(f)
    return np.stack(feats, axis=0)


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n[1/2] Random Forest on HOG features")
    print("    Extracting HOG features...")
    t0 = time.time()
    Xtr_hog = extract_hog_features(X_train)
    Xte_hog = extract_hog_features(X_test)
    print(f"    HOG vector dim = {Xtr_hog.shape[1]}, "
          f"extraction time = {time.time() - t0:.1f}s")

    print("    Training Random Forest (n=200)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=18,
                                 random_state=RNG_SEED, n_jobs=-1)
    rf.fit(Xtr_hog, y_train)
    train_time = time.time() - t0
    train_acc = rf.score(Xtr_hog, y_train)
    test_acc = rf.score(Xte_hog, y_test)
    y_pred = rf.predict(Xte_hog)
    print(f"    Train acc = {train_acc:.3f}  Test acc = {test_acc:.3f}  "
          f"Train time = {train_time:.1f}s")
    return {
        "name": "Random Forest (HOG)",
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "train_time_s": float(train_time),
        "y_pred": y_pred,
        "model": rf,
    }


def train_mlp(X_train, y_train, X_test, y_test):
    print("\n[2/2] Multilayer Perceptron on raw pixels")
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    print(f"    Input dim = {Xtr.shape[1]} (64x64 flattened)")
    print("    Architecture: 4096 -> 256 -> 128 -> 3")
    print("    Training (max 80 epochs, early stopping)...")

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=80,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=10,
                        random_state=RNG_SEED,
                        verbose=False)
    t0 = time.time()
    mlp.fit(Xtr, y_train)
    train_time = time.time() - t0
    train_acc = mlp.score(Xtr, y_train)
    test_acc = mlp.score(Xte, y_test)
    y_pred = mlp.predict(Xte)
    print(f"    Train acc = {train_acc:.3f}  Test acc = {test_acc:.3f}  "
          f"Train time = {train_time:.1f}s ({mlp.n_iter_} epochs)")
    return {
        "name": "MLP (raw pixels)",
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "train_time_s": float(train_time),
        "y_pred": y_pred,
        "model": mlp,
        "loss_curve": list(map(float, mlp.loss_curve_)),
        "val_scores": list(map(float, mlp.validation_scores_)),
        "n_iter": int(mlp.n_iter_),
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_training_curves(mlp_result, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")

    epochs = np.arange(1, len(mlp_result["loss_curve"]) + 1)
    axes[0].plot(epochs, mlp_result["loss_curve"], color="#1f4e79", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("MLP Training Loss")
    axes[0].grid(alpha=0.3)

    val_epochs = np.arange(1, len(mlp_result["val_scores"]) + 1)
    axes[1].plot(val_epochs, mlp_result["val_scores"],
                 color="#c0392b", linewidth=2, label="Validation accuracy")
    axes[1].axhline(mlp_result["test_acc"], color="#27ae60", linestyle="--",
                    label=f"Final test = {mlp_result['test_acc']:.2%}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MLP Validation Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("Figure A — Training Dynamics for the MLP Classifier",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"    saved {save_path.name}")


def plot_confusion(results, y_test, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"], normalize="true")
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"{res['name']}\nTest accuracy = {res['test_acc']:.2%}",
                     fontsize=11)
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Figure B — Normalized Confusion Matrices",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"    saved {save_path.name}")


def plot_predictions(X_test, y_test, results, save_path, n=12):
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(X_test), size=n, replace=False)
    best = results[1] if results[1]["test_acc"] >= results[0]["test_acc"] \
        else results[0]
    y_pred = best["y_pred"]

    fig, axes = plt.subplots(2, 6, figsize=(13, 5), facecolor="white")
    for ax, i in zip(axes.flat, idx):
        ax.imshow(X_test[i], cmap="inferno", origin="lower")
        true_lbl = CLASS_NAMES[y_test[i]]
        pred_lbl = CLASS_NAMES[y_pred[i]]
        correct = (y_test[i] == y_pred[i])
        color = "#27ae60" if correct else "#c0392b"
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}",
                     fontsize=9, color=color)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    plt.suptitle(f"Figure C — Sample Predictions ({best['name']})",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"    saved {save_path.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Galaxy Morphology Classification — Case Study")
    print("=" * 60)

    X, y = load_data()
    print(f"Dataset: {X.shape}, classes = {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RNG_SEED)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    rf_res = train_random_forest(X_train, y_train, X_test, y_test)
    mlp_res = train_mlp(X_train, y_train, X_test, y_test)

    # Detailed reports
    print("\n--- Random Forest classification report ---")
    print(classification_report(y_test, rf_res["y_pred"],
                                 target_names=CLASS_NAMES, digits=3))
    print("\n--- MLP classification report ---")
    print(classification_report(y_test, mlp_res["y_pred"],
                                 target_names=CLASS_NAMES, digits=3))

    # Figures
    print("\nGenerating figures...")
    plot_training_curves(mlp_res, FIG_DIR / "fig_training_curves.png")
    plot_confusion([rf_res, mlp_res], y_test,
                   FIG_DIR / "fig_confusion_matrix.png")
    plot_predictions(X_test, y_test, [rf_res, mlp_res],
                     FIG_DIR / "fig_predictions.png")

    # Persist a JSON summary (excluding non-serialisable model objects)
    summary = {
        "dataset": {
            "total": int(len(y)),
            "per_class_train": {CLASS_NAMES[i]: int((y_train == i).sum())
                                 for i in range(3)},
            "per_class_test": {CLASS_NAMES[i]: int((y_test == i).sum())
                                for i in range(3)},
            "image_size": list(X.shape[1:]),
        },
        "models": [
            {k: v for k, v in r.items() if k not in ("y_pred", "model")}
            for r in (rf_res, mlp_res)
        ],
    }
    summary_path = ROOT / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print("\n=== FINAL ACCURACIES ===")
    for r in (rf_res, mlp_res):
        print(f"  {r['name']:30s}  test acc = {r['test_acc']:.3f}")


if __name__ == "__main__":
    main()
