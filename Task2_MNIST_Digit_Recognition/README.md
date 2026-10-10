# MNIST Digit Recognition

Internship Task 2 - Arch Technologies (Machine Learning).

Recognises handwritten digits (0-9) from 28x28 grayscale images using classic ML models
and a small neural network, then compares them and evaluates the best one.

## Dataset
[MNIST in CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) -
60,000 training and 10,000 test images. Each row = label + 784 pixel values (0-255).
Download `mnist_train.csv` and `mnist_test.csv` and place them in `data/` (not uploaded to GitHub - too large).

## Run
```bash
pip install -r requirements.txt
python mnist_classifier.py --train data/mnist_train.csv --test data/mnist_test.csv
```
Quick test run: add `--max-train 10000`.

## Pipeline
1. Load CSVs, separate labels and pixels
2. Pre-process: scale pixels to 0-1 (PCA to 50 components for KNN and SVM)
3. Train 5 models: Logistic Regression, KNN, Random Forest, SVM (RBF), Neural Network (MLP)
4. Evaluate on the 10,000-image test set: accuracy, macro F1, classification report, confusion matrix
5. Save sample predictions and misclassified examples

## Outputs (`outputs/`)
`model_comparison.csv`, `classification_report.txt`, `confusion_matrix.png`, `sample_digits.png`,
`class_distribution.png`, `sample_predictions.png`, `misclassified_examples.png`, `mnist_model.joblib`
