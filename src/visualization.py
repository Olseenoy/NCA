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
def plot_spc_chart(df: pd.DataFrame, column: str, subgroup_size: int = 1, time_col: str | None = None):
    """
    Enhanced SPC chart with options for I-MR or X-bar/R charts.
    If subgroup_size=1 -> I-MR chart.
    If subgroup_size>1 -> X-bar & R chart.
    """
    data = df[column].dropna().reset_index(drop=True)
    n = len(data)
    x_axis = df[time_col] if time_col and time_col in df.columns else np.arange(n)

    fig = go.Figure()

    if subgroup_size <= 1:
        # Individual / Moving Range chart
        mean_val = data.mean()
        sigma_val = data.std()
        ucl = mean_val + 3*sigma_val
        lcl = mean_val - 3*sigma_val

        fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', name='Values'))
        fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
        fig.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL (+3σ)")
        fig.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL (-3σ)")

        # Highlight points outside control limits
        out_of_control = (data > ucl) | (data < lcl)
        if out_of_control.any():
            fig.add_trace(go.Scatter(
                x=x_axis[out_of_control],
                y=data[out_of_control],
                mode='markers',
                name='Out of Control',
                marker=dict(color='red', size=10, symbol='x')
            ))

    else:
        # X-bar & R chart
        groups = [data[i:i+subgroup_size] for i in range(0, n, subgroup_size)]
        x_bar = [g.mean() for g in groups if len(g)==subgroup_size]
        r_bar = [g.max()-g.min() for g in groups if len(g)==subgroup_size]

        mean_x = np.mean(x_bar)
        sigma_x = np.std(x_bar)
        ucl_x = mean_x + 3*sigma_x
        lcl_x = mean_x - 3*sigma_x

        fig.add_trace(go.Scatter(y=x_bar, mode='lines+markers', name='X-bar'))
        fig.add_hline(y=mean_x, line_dash="dash", line_color="green", annotation_text="X-bar Mean")
        fig.add_hline(y=ucl_x, line_dash="dot", line_color="red", annotation_text="UCL (+3σ)")
        fig.add_hline(y=lcl_x, line_dash="dot", line_color="red", annotation_text="LCL (-3σ)")

        # R chart
        mean_r = np.mean(r_bar)
        sigma_r = np.std(r_bar)
        ucl_r = mean_r + 3*sigma_r
        lcl_r = max(mean_r - 3*sigma_r, 0)

        fig.add_trace(go.Scatter(y=r_bar, mode='lines+markers', name='R (Range)'))
        fig.add_hline(y=mean_r, line_dash="dash", line_color="blue", annotation_text="R Mean")
        fig.add_hline(y=ucl_r, line_dash="dot", line_color="purple", annotation_text="R UCL")
        fig.add_hline(y=lcl_r, line_dash="dot", line_color="purple", annotation_text="R LCL")

    fig.update_layout(
        title=f"SPC Chart for {column}",
        xaxis_title="Sample" if time_col is None else time_col,
        yaxis_title=column
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
