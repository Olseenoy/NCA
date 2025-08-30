# src/clustering.py
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import hdbscan
from config import CLUSTERING_K, RANDOM_STATE, DBSCAN_PARAMS, HDBSCAN_PARAMS

def fit_kmeans(embeddings, k=CLUSTERING_K):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    labels = km.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    return {"algorithm": "KMeans", "model": km, "labels": labels,
            "silhouette": silhouette, "davies_bouldin": davies_bouldin, "k": k}

def fit_dbscan(embeddings, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(embeddings)
    # Ignore -1 noise points for silhouette
    mask = labels != -1
    if mask.sum() > 1:
        silhouette = silhouette_score(embeddings[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(embeddings[mask], labels[mask])
    else:
        silhouette, davies_bouldin = -1, np.inf
    return {"algorithm": "DBSCAN", "model": db, "labels": labels,
            "silhouette": silhouette, "davies_bouldin": davies_bouldin,
            "eps": eps, "min_samples": min_samples}

def fit_hdbscan(embeddings, min_cluster_size=5):
    hb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hb.fit_predict(embeddings)
    mask = labels != -1
    if mask.sum() > 1:
        silhouette = silhouette_score(embeddings[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(embeddings[mask], labels[mask])
    else:
        silhouette, davies_bouldin = -1, np.inf
    return {"algorithm": "HDBSCAN", "model": hb, "labels": labels,
            "silhouette": silhouette, "davies_bouldin": davies_bouldin,
            "min_cluster_size": min_cluster_size}

def evaluate_clustering(embeddings, k_values=None, dbscan_params=None, hdbscan_params=None):
    results = []

    # --- KMeans ---
    if k_values is None:
        k_values = [2, 3, 4, 5, 6, 7]
    for k in k_values:
        results.append(fit_kmeans(embeddings, k))

    # --- DBSCAN ---
    if dbscan_params is None:
        dbscan_params = DBSCAN_PARAMS
    for param in dbscan_params:
        results.append(fit_dbscan(embeddings, **param))

    # --- HDBSCAN ---
    if hdbscan_params is None:
        hdbscan_params = HDBSCAN_PARAMS
    for param in hdbscan_params:
        results.append(fit_hdbscan(embeddings, **param))

    # --- Select best: max silhouette, min Davies-Bouldin ---
    best = max(results, key=lambda x: x["silhouette"] - 0.1 * x["davies_bouldin"])
    return best, results
