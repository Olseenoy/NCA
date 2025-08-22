# src/classification.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def train_text_classifier(X_texts, y_labels, save_path=None):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_texts, y_labels)
    if save_path:
        joblib.dump(pipe, save_path)
    return pipe


def predict(pipe, texts):
    return pipe.predict(texts), pipe.predict_proba(texts)
