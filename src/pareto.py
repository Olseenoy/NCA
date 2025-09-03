# src/pareto.py 
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

def pareto_table(df: pd.DataFrame, category_col: str, weight_col: Optional[str] = None):
    """
    Build Pareto summary table.
    - df: input DataFrame
    - category_col: categorical column to group by
    - weight_col: numerical column to aggregate (defaults to count if None)
    """
    if weight_col is None:
        # just count occurrences
        tab = df[category_col].value_counts().reset_index()
        tab.columns = [category_col, "Count"]
    else:
        tab = df.groupby(category_col)[weight_col].sum().reset_index()
        tab = tab.sort_values(weight_col, ascending=False).reset_index(drop=True)
        tab.rename(columns={weight_col: "Count"}, inplace=True)

    # Sort descending
    tab = tab.sort_values("Count", ascending=False).reset_index(drop=True)

    # Calculate cumulative %
    tab["CumPct"] = tab["Count"].cumsum() / tab["Count"].sum() * 100

    return tab


def pareto_plot(tab: pd.DataFrame, category_col: str):
    """
    Create Pareto chart with bars and cumulative % line.
    """
    fig = go.Figure()

    # Add bar chart
    fig.add_trace(go.Bar(
        x=tab[category_col],
        y=tab["Count"],
        name="Frequency",
        marker_color="steelblue"
    ))

    # Add cumulative % line
    fig.add_trace(go.Scatter(
        x=tab[category_col],
        y=tab["CumPct"],
        name="Cumulative %",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="red")
    ))

    # Layout with 2nd y-axis
    fig.update_layout(
        title="Pareto Chart",
        xaxis=dict(title=category_col),
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 110]
        ),
        bargap=0.2,
        template="plotly_white"
    )

    return fig
