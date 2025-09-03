# src/pareto.py 
from typing import Optional
import pandas as pd
import plotly.graph_objects as go

def pareto_table(df: pd.DataFrame, category_col: str, weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a Pareto summary with columns:
      [<category_col>, 'Count', 'CumPct']
    """
    if weight_col is None:
        tab = df[category_col].value_counts(dropna=False).reset_index()
        tab.columns = [category_col, "Count"]
    else:
        tab = df.groupby(category_col, dropna=False)[weight_col].sum().reset_index()
        tab = tab.rename(columns={weight_col: "Count"})

    tab = tab.sort_values("Count", ascending=False, kind="mergesort").reset_index(drop=True)
    total = tab["Count"].sum()
    tab["CumPct"] = (tab["Count"].cumsum() / (total if total else 1)) * 100
    return tab

def _find_col_case_insensitive(df: pd.DataFrame, wanted: str) -> str:
    for c in df.columns:
        if str(c).lower() == wanted.lower():
            return c
    raise KeyError(f"Column '{wanted}' not found in {list(df.columns)}")

def pareto_plot(tab: pd.DataFrame, category_col: str, weight_col: Optional[str] = None):
    """
    Plot a Pareto chart using bars for Count and a line for cumulative percent.
    Case-insensitive lookup for 'Count' and 'CumPct'.
    """
    # Resolve actual column names (in case of different casing)
    cat_col = _find_col_case_insensitive(tab, category_col)
    count_col = _find_col_case_insensitive(tab, "Count")
    cum_col = _find_col_case_insensitive(tab, "CumPct")

    fig = go.Figure()

    # Bars (frequency)
    fig.add_trace(go.Bar(
        x=tab[cat_col],
        y=tab[count_col],
        name="Frequency"
    ))

    # Cumulative % line (right axis)
    fig.add_trace(go.Scatter(
        x=tab[cat_col],
        y=tab[cum_col],
        name="Cumulative %",
        mode="lines+markers",
        yaxis="y2"
    ))

    fig.update_layout(
        title="Pareto Chart",
        xaxis=dict(title=str(cat_col)),
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 110]
        ),
        bargap=0.2,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

