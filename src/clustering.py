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

    # Calculate multiple metrics
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    # Human-readable interpretation (mainly Silhouette-based)
    interpretation = interpret_clustering_results(silhouette)

    metrics_summary = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "interpretation": interpretation
    }

    return km, labels, metrics_summary

def predict_cluster(km, emb: np.ndarray):
    return km.predict(emb)
