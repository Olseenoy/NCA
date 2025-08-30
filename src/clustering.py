# src/clustering.py
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from config import CLUSTERING_K, RANDOM_STATE, DBSCAN_PARAMS, HDBSCAN_PARAMS

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
✅ Changes made:

evaluate_clustering() now accepts dbscan_params and hdbscan_params.

Each algorithm computes silhouette and Davies-Bouldin (ignoring noise points for DBSCAN/HDBSCAN).

Returns best clustering automatically for Streamlit visualization.

Supports multiple k-values and parameter combinations.

If you like, I can also patch your streamlit_app.py so that clustering now runs all three algorithms automatically and keeps the visualization persistent throughout other analyses. This will align perfectly with your Key Features.

Do you want me to do that next?







You said:
okay


ChatGPT said:
Perfect! Let’s patch your streamlit_app.py clustering section to:

Run KMeans, DBSCAN, and HDBSCAN automatically.

Select the best clustering based on silhouette + Davies-Bouldin.

Keep the visualization persistent across Pareto, SPC, Trend, and RCA analyses.

Here’s the upgraded section:

python
Copy code
# --- Clustering & Visualization ---
st.subheader("Clustering & Visualization")

if valid_p and valid_embeddings:
    if st.button('Run Clustering'):
        try:
            from clustering import evaluate_clustering
            with st.spinner("Evaluating KMeans, DBSCAN, and HDBSCAN..."):
                best, all_results = evaluate_clustering(
                    embeddings,
                    k_values=list(range(2, 8)),
                    dbscan_params=[{"eps": 0.5, "min_samples": 5}, {"eps": 0.7, "min_samples": 5}],
                    hdbscan_params=[{"min_cluster_size": 5}, {"min_cluster_size": 10}]
                )

            # Save results to session state
            st.session_state['cluster_best'] = best
            st.session_state['cluster_all'] = all_results
            st.session_state['cluster_fig'] = cluster_scatter(embeddings, best["labels"])
            st.session_state['cluster_text'] = (
                f"Best: {best['algorithm']} | K={best.get('k', '-')}, "
                f"Silhouette={best['silhouette']:.3f}, Davies-Bouldin={best['davies_bouldin']:.3f}"
            )

        except Exception as e:
            st.error(f"Clustering failed: {e}")

# Display persistent clustering visualization
if 'cluster_fig' in st.session_state:
    st.success(st.session_state['cluster_text'])
    st.plotly_chart(st.session_state['cluster_fig'], use_container_width=True)


