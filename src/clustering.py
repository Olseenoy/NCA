# src/clustering.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from config import CLUSTERING_K, RANDOM_STATE


def fit_kmeans(embeddings: np.ndarray, k: int = CLUSTERING_K):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    return km, labels, score


def predict_cluster(km, emb: np.ndarray):
    return km.predict(emb)
