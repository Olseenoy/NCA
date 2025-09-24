# src/visualization.py
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np 
from typing import Dict, List, Tuple

# --- Pareto Chart ---

def pareto_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Create a Pareto table: categories, counts, %, cumulative %.
    Works with text or numeric columns.
    """
    if column not in df.columns:
        return pd.DataFrame()

    # Drop NaN and convert everything to string (so categories are uniform)
    series = df[column].dropna().astype(str)

    if series.empty:
        return pd.DataFrame()

    counts = series.value_counts()
    total = counts.sum()

    tab = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
    })
    tab["Percent"] = (tab["Count"] / total * 100).round(2)
    tab["Cumulative %"] = tab["Percent"].cumsum().round(2)

    return tab


def pareto_plot(tab: pd.DataFrame) -> go.Figure | None:
    """Plot Pareto chart with bar (counts) + line (cumulative %)."""
    if tab is None or tab.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tab["Category"], y=tab["Count"], name="Count", yaxis="y"))
    fig.add_trace(go.Scatter(
        x=tab["Category"], y=tab["Cumulative %"],
        name="Cumulative %", yaxis="y2", mode="lines+markers"
    ))

    fig.update_layout(
        title="Pareto Chart",
        xaxis=dict(title="Category", tickangle=-45),
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        bargap=0.2,
        margin=dict(t=50, b=150)
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


# ------------------------------
# Time-Series Plot Function
# ------------------------------
def plot_time_series_trend(df, time_col, value_col, freq="D", agg_func="mean"):
    """
    Plot time-series trends with aggregation.

    freq: "D" (daily), "W" (weekly), "M" (monthly), "Y" (yearly)
    agg_func: "mean", "sum", "max", "min"
    """
    try:
        temp = df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
        temp = temp.dropna(subset=[time_col, value_col])

        # Resample with chosen aggregation
        if agg_func == "sum":
            agg_df = temp.set_index(time_col).resample(freq)[value_col].sum().reset_index()
        elif agg_func == "max":
            agg_df = temp.set_index(time_col).resample(freq)[value_col].max().reset_index()
        elif agg_func == "min":
            agg_df = temp.set_index(time_col).resample(freq)[value_col].min().reset_index()
        else:  # default mean
            agg_df = temp.set_index(time_col).resample(freq)[value_col].mean().reset_index()

        fig = px.line(
            agg_df,
            x=time_col,
            y=value_col,
            title=f"{value_col} Trend ({freq}, {agg_func})",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col,
            template="plotly_white",
            hovermode="x unified"
        )
        return fig
    except Exception as e:
        st.error(f"Time-series plot failed: {e}")
        return None
        
# snca_rca_module.py

def rule_based_rca_fallback(issue_text, processed_df=None):
    """
    Very simple rule-based RCA fallback.
    Scans processed data for patterns and generates generic RCA.
    """
    root_causes = ["Insufficient data for AI RCA. Used fallback."]
    five_whys = [
        f"Why 1: {issue_text} occurred due to lack of clear cause.",
        "Why 2: No deeper analysis available without AI.",
        "Why 3: Using rule-based template.",
        "Why 4: Limited context from data.",
        "Why 5: Suggest collecting more structured RCA cases.",
    ]
    capa = [
        {"type": "Preventive", "action": "Collect more RCA cases", "owner": "QA", "due_in_days": 30},
        {"type": "Corrective", "action": "Investigate issue manually", "owner": "Ops", "due_in_days": 7},
    ]

    fishbone = {
        "Man": ["Insufficient training"],
        "Machine": ["No AI inference available"],
        "Method": ["Fallback rule-based analysis"],
        "Material": [],
        "Measurement": [],
        "Environment": [],
    }

    return {
        "analysis": "Fallback RCA used. AI analysis failed or unavailable.",
        "root_causes": root_causes,
        "five_whys": five_whys,
        "capa": capa,
        "fishbone": fishbone,
    }


def visualize_fishbone_plotly(fishbone_data):
    """
    Plot fishbone diagram using Plotly.
    """
    fig = go.Figure()

    categories = list(fishbone_data.keys())
    y_positions = list(range(len(categories)))

    for y, cat in zip(y_positions, categories):
        causes = fishbone_data.get(cat, [])
        for c in causes:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[y, y],
                    mode="lines+text",
                    text=[cat, c],
                    textposition="top center"
                )
            )

    fig.update_layout(
        title="Fishbone Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
    )

    return fig




