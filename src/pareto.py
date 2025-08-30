# --- Pareto Chart ---
import plotly.graph_objects as go
import pandas as pd

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
