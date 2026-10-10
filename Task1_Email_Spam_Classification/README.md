# Email Spam Classification

Internship Task 1 - Arch Technologies (Machine Learning).

Classifies emails as **spam** or **not spam** using TF-IDF features and classic ML models
(Naive Bayes, Logistic Regression, Linear SVM). The best model (by F1-score) is evaluated and saved.

## Dataset
[Spam Email Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) - ~83k emails (TREC 2007 + Enron-Spam) with `label` (1 = spam, 0 = not spam) and `text` columns. Download the CSV and place it in `data/`.

## Run
```bash
pip install -r requirements.txt
python spam_classifier.py --data data/combined_data.csv
```
If your columns are not auto-detected: `--text-col <name> --label-col <name>`.

## Pipeline
1. Load data, convert labels to 0/1, drop duplicates
2. Clean text (lowercase, replace URLs/emails/numbers, strip symbols)
3. TF-IDF (unigrams + bigrams, stop-words removed)
4. Train and compare 3 models on an 80/20 stratified split
5. Evaluate with accuracy, precision, recall, F1, confusion matrix

## Outputs (`outputs/`)
`model_comparison.csv`, `classification_report.txt`, `confusion_matrix.png`,
`class_distribution.png`, `spam_model.joblib`
