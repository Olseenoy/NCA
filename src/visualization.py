# src/visualization.py
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np 

# --- Pareto Chart ---
# --- Pareto Chart ---
def pareto_plot(pareto_df: pd.DataFrame):
    """
    Generates a Pareto chart with bars and cumulative line.
    """
    fig = px.bar(
        pareto_df.reset_index(),
        x=pareto_df.index,
        y='count',
        labels={'x': 'category', 'count': 'count'},
        title="Pareto Analysis"
    )
    
    # Use np.array to ensure compatibility
    cumulative_y = np.array(pareto_df['cum_pct'] * pareto_df['count'].sum())
    
    fig.add_scatter(
        x=pareto_df.index,
        y=cumulative_y,
        yaxis='y2',
        name='Cumulative',
        mode='lines+markers'
    )
    
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            title='Cumulative'
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig
# --- Clustering Scatter Plot ---
def cluster_scatter(embeddings, labels):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(embeddings)
    df = pd.DataFrame({'x': pts[:, 0], 'y': pts[:, 1], 'cluster': labels})
    fig = px.scatter(
        df, x='x', y='y', color='cluster',
        hover_data=['cluster'],
        title="Cluster Visualization (PCA Projection)"
    )
    fig.update_layout(height=600)  # Keep chart large and visible
    return fig

# --- Clustering Metrics Visualization ---
def clustering_metrics_chart(metrics_summary):
    """
    Visualizes clustering quality metrics in a bar chart.
    """
    metrics = {k: v for k, v in metrics_summary.items() if k != "interpretation"}
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])

    fig = px.bar(
        df, x='Metric', y='Score', text='Score',
        title="Clustering Quality Metrics",
        color='Metric'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1], height=400)
    return fig

# --- SPC Chart ---
def plot_spc_chart(df, column):
    """
    Simple SPC (Statistical Process Control) chart with mean and ±3σ limits.
    """
    data = df[column].dropna().reset_index(drop=True)
    mean_val = data.mean()
    std_val = data.std()

    ucl = mean_val + 3 * std_val
    lcl = mean_val - 3 * std_val

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name=column))
    fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL (+3σ)")
    fig.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL (-3σ)")
    fig.update_layout(title=f"SPC Chart for {column}", xaxis_title="Sample", yaxis_title=column)
    return fig

def plot_trend_dashboard(df, date_col, value_col):
    if date_col not in df.columns or value_col not in df.columns:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    df = df.sort_values(date_col)
    fig = px.line(df, x=date_col, y=value_col, title="Trend Dashboard")
    return fig

def plot_time_series_trend(df, date_col, value_col):
    if date_col not in df.columns or value_col not in df.columns:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    df = df.sort_values(date_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines+markers'))
    fig.update_layout(title="Time Series Trend", xaxis_title=date_col, yaxis_title=value_col)
    return fig
