# Multi-Label Emotion Classification using TF-IDF and Logistic Regression

This project performs **multi-label emotion classification** on Reddit comments using TF-IDF features and a logistic regression model trained in a One-vs-Rest setup.

## 📁 Files

- `train.tsv` – Training data (`text`, `labels`, `id`)
- `test.tsv` – Testing data (`text`, `labels`, `id`)
- `emotions.txt` – List of 28 emotion labels, one per line
- Python script – Main code for preprocessing, training, evaluation, and prediction

## 📦 Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
