from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from config import CLUSTERING_K, RANDOM_STATE

def interpret_clustering_results(score: float) -> str:
    """
    Returns a human-readable interpretation of the silhouette score.
    """
    if score < 0.2:
        return (
            f"The silhouette score of {score:.3f} indicates very weak clustering. "
            "Clusters overlap significantly; consider increasing data quality, "
            "adding more relevant features, or changing the number of clusters."
        )
    elif score < 0.5:
        return (
            f"The silhouette score of {score:.3f} suggests moderate clustering. "
            "Some cluster separation exists, but improvements can be made. "
            "Experiment with different K values or embeddings."
        )
    elif score < 0.7:
        return (
            f"The silhouette score of {score:.3f} indicates good clustering. "
            "Clusters are reasonably well-separated with minimal overlap."
        )
    else:
        return (
            f"Excellent clustering with a silhouette score of {score:.3f}. "
            "Clusters are highly distinct, and separation is very strong."
        )

def fit_kmeans(embeddings: np.ndarray, k: int = CLUSTERING_K):
    """
    Fit KMeans and return clustering results along with multiple metrics.
    """
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    labels = km.fit_predict(embeddings)

    # Calculate metrics
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    # Human-readable interpretation
    interpretation = interpret_clustering_results(silhouette)

    metrics_summary = {
        "k": k,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "interpretation": interpretation
    }

    return km, labels, metrics_summary

def evaluate_kmeans(embeddings: np.ndarray, k_values=None):
    """
    Test multiple K values and select best one based on silhouette & Davies-Bouldin.
    """
    if k_values is None:
        k_values = range(2, 8)  # default: test K=2 to K=7

    results = []
    for k in k_values:
        km, labels, metrics = fit_kmeans(embeddings, k)
        results.append({**metrics, "labels": labels, "km": km})

    # Select best result: highest silhouette and lowest Davies-Bouldin
    best = max(results, key=lambda x: x["Silhouette Score"] - 0.1 * x["Davies-Bouldin Score"])
    return best, results

def predict_cluster(km, emb: np.ndarray):
    return km.predict(emb)
