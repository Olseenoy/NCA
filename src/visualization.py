# src/visualization.py
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA


def pareto_plot(pareto_df: pd.DataFrame):
    fig = px.bar(pareto_df.reset_index(), x=pareto_df.index, y='count', labels={'x':'category','count':'count'})
    fig.add_scatter(x=pareto_df.index, y=pareto_df['cum_pct']*pareto_df['count'].sum(), yaxis='y2', name='cumulative')
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Cumulative'))
    return fig


def cluster_scatter(embeddings, labels):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(embeddings)
    df = pd.DataFrame({'x': pts[:,0], 'y': pts[:,1], 'cluster': labels})
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['cluster'])
    return fig
