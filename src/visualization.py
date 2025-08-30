# src/visualization.py
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np 

# --- Pareto Chart ---
def pareto_plot(pareto_df: pd.DataFrame, title: str = "Pareto Chart"):
    """
    Creates a Pareto chart with bars (counts) and a cumulative percentage line.
    
    Expects pareto_df to have:
    - 'count': the absolute count for each category
    - 'cum_pct': cumulative percentage (0-1)
    """
    if not isinstance(pareto_df, pd.DataFrame):
        raise ValueError("pareto_df must be a pandas DataFrame")

    if 'count' not in pareto_df.columns or 'cum_pct' not in pareto_df.columns:
        raise ValueError("pareto_df must contain 'count' and 'cum_pct' columns")

    categories = pareto_df.index.astype(str)
    counts = pareto_df['count'].values
    cum_pct = pareto_df['cum_pct'].values * 100  # convert to percentage

    fig = go.Figure()

    # Bar for counts
    fig.add_trace(go.Bar(
        x=categories,
        y=counts,
        name='Count',
        marker_color='steelblue',
        text=counts,
        textposition='auto',
        yaxis='y1',
        hovertemplate='Category: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Line for cumulative %
    fig.add_trace(go.Scatter(
        x=categories,
        y=cum_pct,
        name='Cumulative %',
        mode='lines+markers',
        line=dict(color='crimson', width=2),
        yaxis='y2',
        hovertemplate='Category: %{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis=dict(title="Count", showgrid=True),
        yaxis2=dict(
            title="Cumulative %",
            overlaying='y',
            side='right',
            range=[0, 110]
        ),
        legend=dict(x=0.8, y=1.1),
        template="plotly_white",
        margin=dict(t=50, b=50)
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
    Enhanced SPC (Statistical Process Control) chart with mean, ±3σ limits,
    and highlighting of out-of-control points.
    """
    data = df[column].dropna().reset_index(drop=True)
    mean_val = data.mean()
    std_val = data.std()

    ucl = mean_val + 3 * std_val
    lcl = mean_val - 3 * std_val

    # Highlight points outside control limits
    colors = ['red' if v > ucl or v < lcl else 'blue' for v in data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data, mode='lines+markers',
        name=column,
        marker=dict(color=colors, size=10)
    ))

    # Control lines
    fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL (+3σ)")
    fig.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL (-3σ)")

    fig.update_layout(
        title=f"SPC Chart for {column}",
        xaxis_title="Sample",
        yaxis_title=column,
        template="plotly_white",
        showlegend=True
    )
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
