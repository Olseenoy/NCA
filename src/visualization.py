# src/visualization.py
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
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

# d3 format mapping for Plotly tickformat (keeps simple 1:1 mapping where possible)
D3_FORMATS = {
    "%Y-%m-%d": "%Y-%m-%d",   # 2025-09-27
    "%d-%m-%Y": "%d-%m-%Y",   # 27-09-2025
    "%m-%d-%Y": "%m-%d-%Y",   # 09-27-2025
    "%d/%m/%Y": "%d/%m/%Y",   # 27/09/2025
    "%m/%d/%Y": "%m/%d/%Y",   # 09/27/2025
    "%Y/%m/%d": "%Y/%m/%d",   # 2025/09/27
    "%b %Y": "%b %Y",         # Sep 2025
    "%d %b %Y": "%d %b %Y",   # 27 Sep 2025
    "%b %d, %Y": "%b %d, %Y", # Sep 27, 2025
    "%H:%M": "%H:%M",         # 14:05
    "%H:%M:%S": "%H:%M:%S",   # 14:05:33
}

def _clean_date_strings(series):
    s = series.astype(str).str.strip()
    # replace common unicode dashes with ASCII hyphen, remove weird invisible chars
    s = s.str.replace("\u2013", "-", regex=False).str.replace("\u2014", "-", regex=False)
    s = s.str.replace(r"\u00A0", "", regex=True)  # remove non-breaking spaces
    return s

def parse_dates_strict(series, date_format=None):
    """
    Strictly parse according to date_format (if provided).
    Returns a datetime64[ns] Series and diagnostics dict.
    """
    s_clean = _clean_date_strings(series)
    if date_format:
        parsed = pd.to_datetime(s_clean, format=date_format, errors="coerce")
    else:
        parsed = pd.to_datetime(s_clean, errors="coerce")
    diagnostics = {
        "total": len(parsed),
        "parsed_count": int(parsed.notna().sum()),
        "na_count": int(parsed.isna().sum()),
        "sample_raw": s_clean.head(10).tolist(),
        "sample_parsed": parsed.head(10).astype(str).tolist(),
        "failed_examples": s_clean[parsed.isna()].head(10).tolist()
    }
    return parsed, diagnostics

def plot_trend_dashboard(df, date_col, value_col, date_format=None):
    df_plot = df.copy()

    # If a date_format is provided, add a formatted string version for plotting axis labels
    if date_format:
        try:
            df_plot["_formatted_date"] = df_plot[date_col].dt.strftime(date_format)
            x_col = "_formatted_date"
        except Exception:
            x_col = date_col  # fallback to raw datetime
    else:
        x_col = date_col

    fig = px.line(df_plot, x=x_col, y=value_col,
                  title=f"Trend of {value_col} over {date_col}")

    # Ensure x-axis is treated correctly
    fig.update_xaxes(type="category" if date_format else "date")

    fig.update_layout(xaxis_title=date_col, yaxis_title=value_col)
    return fig




def plot_time_series_trend(df, date_col, value_col, freq="D", agg_func="mean", date_format=None):
    # df[date_col] must be datetime and not-null rows will be used
    # Drop NA datetimes then set index and resample
    tmp = df[[date_col, value_col]].copy()
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return None
    tmp = tmp.sort_values(by=date_col)
    tmp = tmp.set_index(date_col)
    # resample + agg
    try:
        resampled = tmp[value_col].resample(freq).agg(agg_func).reset_index()
    except Exception as e:
        # if resample fails, return None so caller can show a message
        raise RuntimeError(f"resample failed: {e}")
    fig = px.line(resampled, x=date_col, y=value_col, title=f"{agg_func.capitalize()} {value_col} ({freq})")
    fig.update_xaxes(type="date")
    if date_format:
        fig.update_xaxes(tickformat=D3_FORMATS.get(date_format, "%Y-%m-%d"))
    fig.update_layout(xaxis_title=date_col, yaxis_title=value_col)
    return fig


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
    Draws a fishbone (Ishikawa) diagram from categorized root causes.
    fishbone_data = {
        "Machine": ["Poor maintenance", "Wear & tear"],
        "Method": ["Incorrect sealing procedure"],
        "Material": ["Seal quality issues"],
        "Manpower": ["Operator error"],
        "Measurement": ["Improper calibration"],
        "Environment": ["High temperature", "Humidity"]
    }
    """

    fig = go.Figure()

    # Draw backbone
    fig.add_shape(type="line", x0=0.1, y0=0.5, x1=0.9, y1=0.5,
                  line=dict(color="black", width=3))

    # Position categories alternately top/bottom
    categories = list(fishbone_data.keys())
    spacing = 0.8 / (len(categories) - 1)
    y_offsets = [0.7, 0.3] * ((len(categories) // 2) + 1)

    for i, cat in enumerate(categories):
        x = 0.15 + i * spacing
        y = 0.5
        y_target = y_offsets[i]

        # Draw main bone
        fig.add_shape(type="line", x0=x, y0=y, x1=x+0.1, y1=y_target,
                      line=dict(color="black", width=2))

        # Add category label
        fig.add_annotation(x=x+0.12, y=y_target, text=cat,
                           showarrow=False, font=dict(size=12, color="blue"))

        # Add sub-causes as bullet points
        causes = fishbone_data[cat]
        for j, cause in enumerate(causes):
            fig.add_annotation(
                x=x+0.15, y=y_target + (0.05 if y_target > y else -0.05) * (j+1),
                text=f"• {cause}", showarrow=False,
                font=dict(size=10, color="black"), align="left"
            )

    fig.update_layout(
        title="Fishbone Diagram (Ishikawa)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=600
    )

    return fig





