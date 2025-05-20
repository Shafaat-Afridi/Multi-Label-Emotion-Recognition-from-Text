import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# === Load label names from emotions.txt ===
with open('emotions.txt', 'r') as f:
    emotion_labels = [line.strip() for line in f.readlines()]

# === Load train and test data ===
train_df = pd.read_csv('train.tsv', sep='\t', names=['text', 'labels', 'ids'])
test_df = pd.read_csv('test.tsv', sep='\t', names=['text', 'labels', 'ids'])

# === Combine train and test for consistent label binarization ===
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# === Convert string label lists to integer lists ===
def parse_labels(label_str):
    return list(map(int, label_str.split(',')))

combined_df['label_list'] = combined_df['labels'].apply(parse_labels)

# === Binarize labels ===
mlb = MultiLabelBinarizer(classes=list(range(len(emotion_labels))))
Y_all = mlb.fit_transform(combined_df['label_list'])

# === Prepare TF-IDF vectors ===
vectorizer = TfidfVectorizer(max_features=10000)
X_all = vectorizer.fit_transform(combined_df['text'])

# === Split back into train and test ===
X_train = X_all[:len(train_df)]
X_test = X_all[len(train_df):]
y_train = Y_all[:len(train_df)]
y_test = Y_all[len(train_df):]

# === Train classifier ===
clf = OneVsRestClassifier(LogisticRegression(max_iter=300))
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=emotion_labels))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Predict on new text ===
def predict_emotions(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)
    return [emotion_labels[i] for i, val in enumerate(pred[0]) if val == 1]

# Example
text = "I’m so excited and happy for you!"
print("\nInput:", text)
print("Predicted Emotions:", predict_emotions(text))
