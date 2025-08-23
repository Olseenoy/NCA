# src/visualization.py
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# --- Pareto Chart ---
def pareto_plot(pareto_df: pd.DataFrame):
    fig = px.bar(
        pareto_df.reset_index(),
        x=pareto_df.index,
        y='count',
        labels={'x': 'category', 'count': 'count'}
    )
    fig.add_scatter(
        x=pareto_df.index,
        y=pareto_df['cum_pct'] * pareto_df['count'].sum(),
        yaxis='y2',
        name='cumulative'
    )
    fig.update_layout(
        yaxis2=dict(overlaying='y', side='right', title='Cumulative')
    )
    return fig

# --- Clustering Scatter Plot ---
def cluster_scatter(embeddings, labels):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(embeddings)
    df = pd.DataFrame({'x': pts[:, 0], 'y': pts[:, 1], 'cluster': labels})
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['cluster'])
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
