"""
MNIST Digit Recognition - Arch Technologies Internship (Task 2)

Works with the Kaggle "MNIST in CSV" dataset (mnist_train.csv / mnist_test.csv)
and also with the Kaggle "Digit Recognizer" train.csv.

Usage:
    python mnist_classifier.py --train data/mnist_train.csv --test data/mnist_test.csv
    python mnist_classifier.py --train data/mnist_train.csv --test data/mnist_test.csv --max-train 10000   (quick run)
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
OUT = Path("outputs")
SEED = 42


# ---------- 1. Load ----------
def load_xy(path):
    """Return (X, y) from a CSV with a 'label' column + 784 pixel columns.
    Returns (X, None) if the file has no label column (e.g. Kaggle test.csv)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "label" not in cols:
        return df.values.astype(np.float32), None
    y = df[cols["label"]].values.astype(int)
    X = df.drop(columns=[cols["label"]]).values.astype(np.float32)
    return X, y


# ---------- 2. Pre-process ----------
def preprocess(X):
    """Scale pixel values from 0-255 to 0-1."""
    return X / 255.0


# ---------- Plot helpers ----------
def plot_samples(X, y, path):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for digit, ax in enumerate(axes.ravel()):
        idx = np.where(y == digit)[0][0]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Digit {digit}"); ax.axis("off")
    plt.suptitle("Sample MNIST Images (one per class)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_class_distribution(y, path):
    counts = pd.Series(y).value_counts().sort_index()
    counts.plot(kind="bar", color="#4C72B0", rot=0)
    plt.title("Training Set Class Distribution")
    plt.xlabel("Digit"); plt.ylabel("Images")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_predictions(X, y_true, y_pred, path, title, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    for ax, i in zip(axes.ravel(), range(n)):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}", color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ---------- 3. Models ----------
def build_models(n_jobs):
    return {
        "Logistic Regression": Pipeline([
            ("clf", LogisticRegression(max_iter=100))]),
        "KNN (PCA-50)": Pipeline([
            ("pca", PCA(n_components=50, random_state=SEED)),
            ("clf", KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs))]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, n_jobs=n_jobs,
                                           random_state=SEED))]),
        "SVM RBF (PCA-50)": Pipeline([
            ("pca", PCA(n_components=50, random_state=SEED)),
            ("clf", SVC(kernel="rbf", C=5, gamma="scale"))]),
        "Neural Network (MLP)": Pipeline([
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30,
                                  early_stopping=True, random_state=SEED))]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", help="optional labelled test CSV; otherwise 80/20 split")
    ap.add_argument("--max-train", type=int, help="use only N training images (quick run)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    X_train, y_train = load_xy(args.train)
    if y_train is None:
        raise SystemExit("Training file needs a 'label' column.")
    X_test = y_test = None
    if args.test:
        X_test, y_test = load_xy(args.test)
    if y_test is None:  # no usable test file -> split the training data
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    if args.max_train and args.max_train < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=args.max_train, random_state=SEED, stratify=y_train)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    plot_samples(X_train, y_train, OUT / "sample_digits.png")
    plot_class_distribution(y_train, OUT / "class_distribution.png")

    X_train, X_test = preprocess(X_train), preprocess(X_test)

    # Train + compare
    results, fitted = [], {}
    for name, model in build_models(args.n_jobs).items():
        print(f"Training {name} ...", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append({"Model": name, "Accuracy": acc,
                        "F1 (macro)": f1_score(y_test, pred, average="macro"),
                        "Train Time (s)": train_time})
        fitted[name] = model
        print(f"   accuracy = {acc:.4f}  ({train_time:.0f}s)", flush=True)

    res = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    res[["Accuracy", "F1 (macro)"]] = res[["Accuracy", "F1 (macro)"]].round(4)
    res["Train Time (s)"] = res["Train Time (s)"].round(1)
    print("\nModel comparison:\n", res.to_string(index=False))
    res.to_csv(OUT / "model_comparison.csv", index=False)

    # Evaluate best model
    best_name = res.iloc[0]["Model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, digits=4)
    print(f"\nBest model: {best_name}\n{report}")
    (OUT / "classification_report.txt").write_text(f"Best model: {best_name}\n\n{report}")

    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    # Sample predictions and mistakes
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_test), 10, replace=False)
    plot_predictions(X_test[idx], y_test[idx], pred[idx],
                     OUT / "sample_predictions.png",
                     f"Random Test Predictions - {best_name}")
    wrong = np.where(pred != y_test)[0][:10]
    if len(wrong):
        plot_predictions(X_test[wrong], y_test[wrong], pred[wrong],
                         OUT / "misclassified_examples.png",
                         "Misclassified Examples", n=len(wrong))

    # Save model (skip if too big for GitHub's 100 MB limit)
    model_path = OUT / "mnist_model.joblib"
    joblib.dump(best, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6
    if size_mb > 90:
        model_path.unlink()
        print(f"Model was {size_mb:.0f} MB (too big for GitHub) - not saved.")
    else:
        print(f"Saved model ({size_mb:.1f} MB) -> {model_path}")


if __name__ == "__main__":
    main()
