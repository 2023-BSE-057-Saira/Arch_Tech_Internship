"""
Email Spam Classification - Arch Technologies Internship (Task 1)

Usage:
    python spam_classifier.py --data data/spam.csv
    python spam_classifier.py --data data/emails.csv --text-col text --label-col spam
"""
import argparse
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

OUT = Path("outputs")
TEXT_CANDIDATES = ["text", "v2", "message", "email", "body", "content"]
LABEL_CANDIDATES = ["label", "v1", "spam", "category", "class", "target"]


# ---------- 1. Load ----------
def load_data(path, text_col=None, label_col=None):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:  # the classic spam.csv needs latin-1
        df = pd.read_csv(path, encoding="latin-1")

    lower = {c.lower(): c for c in df.columns}
    text_col = text_col or next((lower[c] for c in TEXT_CANDIDATES if c in lower), None)
    label_col = label_col or next((lower[c] for c in LABEL_CANDIDATES if c in lower), None)
    if text_col is None or label_col is None:
        raise SystemExit(f"Could not detect columns. Found: {list(df.columns)}. "
                         "Pass --text-col and --label-col.")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Convert labels to 0 (not spam) / 1 (spam)
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            lambda x: 1 if x == "spam" else 0)
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


# ---------- 2. Pre-process ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)      # links
    text = re.sub(r"\S+@\S+", " emailaddr ", text)          # email addresses
    text = re.sub(r"\d+", " num ", text)                    # numbers
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation/symbols
    return re.sub(r"\s+", " ", text).strip()


# ---------- 3. Models ----------
def build_models():
    def tfidf():
        # max_features keeps memory/time reasonable on large datasets (80k+ emails)
        return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                               min_df=3, max_df=0.95, max_features=50000,
                               sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", LinearSVC())]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = load_data(args.data, args.text_col, args.label_col)
    print(f"Dataset: {len(df)} emails | spam ratio: {df['label'].mean():.2%}")

    # Class distribution plot
    df["label"].map({0: "Not Spam", 1: "Spam"}).value_counts().plot(
        kind="bar", color=["#4C72B0", "#C44E52"], rot=0)
    plt.title("Class Distribution"); plt.ylabel("Emails"); plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150); plt.close()

    df["clean"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    # Train + compare
    results, fitted = [], {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        fitted[name] = model
    res = pd.DataFrame(results).sort_values("F1", ascending=False).round(4)
    print("\nModel comparison:\n", res.to_string(index=False))
    res.to_csv(OUT / "model_comparison.csv", index=False)

    # Evaluate best model
    best_name = res.iloc[0]["Model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, target_names=["Not Spam", "Spam"])
    print(f"\nBest model: {best_name}\n{report}")
    (OUT / "classification_report.txt").write_text(f"Best model: {best_name}\n\n{report}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Not Spam", "Spam"], cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}"); plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    joblib.dump(best, OUT / "spam_model.joblib")

    # Demo predictions (good screenshot material)
    samples = [
        "Congratulations! You won a $1000 gift card. Click http://claim-now.xyz to redeem",
        "Hi team, the meeting is moved to 3pm tomorrow. Please review the agenda.",
        "URGENT: your account is suspended, verify your password immediately",
    ]
    print("\nDemo predictions:")
    for s in samples:
        label = "SPAM" if best.predict([clean_text(s)])[0] == 1 else "NOT SPAM"
        print(f"[{label}] {s}")


if __name__ == "__main__":
    main()
