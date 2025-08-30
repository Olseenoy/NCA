# src/clustering.py
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from config import CLUSTERING_K, RANDOM_STATE

# Optional: only if you installed hdbscan
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

def interpret_clustering_results(score: float) -> str:
    """Return a human-readable interpretation of silhouette score."""
    if score < 0.2:
        return f"Silhouette={score:.3f} indicates very weak clustering. Consider better features or changing parameters."
    elif score < 0.5:
        return f"Silhouette={score:.3f} suggests moderate clustering. Some cluster separation exists."
    elif score < 0.7:
        return f"Silhouette={score:.3f} indicates good clustering. Clusters are reasonably well-separated."
    else:
        return f"Silhouette={score:.3f} indicates excellent clustering. Clusters are highly distinct."

def fit_kmeans(embeddings: np.ndarray, k: int = CLUSTERING_K):
    """Fit KMeans and calculate metrics."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    labels = km.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    metrics = {
        "method": "KMeans",
        "k": k,
        "labels": labels,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "interpretation": interpret_clustering_results(silhouette),
    }
    return metrics

def fit_dbscan(embeddings: np.ndarray, eps=0.5, min_samples=5):
    """Fit DBSCAN and calculate metrics (ignore noise points for silhouette)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(embeddings)
    # check if at least 2 clusters exist
    unique_labels = set(labels)
    if len(unique_labels - {-1}) <= 1:
        silhouette = -1
        davies_bouldin = np.inf
    else:
        mask = labels != -1  # exclude noise
        silhouette = silhouette_score(embeddings[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(embeddings[mask], labels[mask])
    metrics = {
        "method": "DBSCAN",
        "labels": labels,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "interpretation": interpret_clustering_results(silhouette),
    }
    return metrics

def fit_hdbscan(embeddings: np.ndarray, min_cluster_size=5):
    """Fit HDBSCAN and calculate metrics (ignore noise points)."""
    if not HDBSCAN_AVAILABLE:
        return None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    unique_labels = set(labels)
    if len(unique_labels - {-1}) <= 1:
        silhouette = -1
        davies_bouldin = np.inf
    else:
        mask = labels != -1
        silhouette = silhouette_score(embeddings[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(embeddings[mask], labels[mask])
    metrics = {
        "method": "HDBSCAN",
        "labels": labels,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "interpretation": interpret_clustering_results(silhouette),
    }
    return metrics

def evaluate_clustering(embeddings: np.ndarray, k_values=None):
    """Run KMeans, DBSCAN, HDBSCAN and select the best based on metrics."""
    if k_values is None:
        k_values = range(2, 8)
    results = []

    # KMeans
    for k in k_values:
        results.append(fit_kmeans(embeddings, k))

    # DBSCAN
    results.append(fit_dbscan(embeddings))

    # HDBSCAN (optional)
    hdb = fit_hdbscan(embeddings)
    if hdb:
        results.append(hdb)

    # Select best: highest silhouette minus Davies-Bouldin penalty
    def score_fn(x):
        return x["Silhouette Score"] - 0.1 * x["Davies-Bouldin Score"]

    best = max(results, key=score_fn)
    return best, results

def predict_cluster(model_labels: np.ndarray, new_embeddings: np.ndarray):
    """Predict clusters (only works for KMeans for now)."""
    return model_labels


