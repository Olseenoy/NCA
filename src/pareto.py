# src/pareto.py 
import pandas as pd 
from typing import Optional




def pareto_table(df: pd.DataFrame, category_col: str, weight_col: str | None = None):
    if weight_col:
        s = df.groupby(category_col)[weight_col].sum().sort_values(ascending=False)
    else:
        s = df[category_col].value_counts()

    cum = s.cumsum() / s.sum() * 100  # convert to percentage
    tab = pd.DataFrame({
        category_col: s.index,
        "count": s.values,
        "cum_pct": cum.values
    })

    return tab.reset_index(drop=True)


def pareto_plot(pareto_df: pd.DataFrame):
    # detect category column (first column of df)
    category_col = pareto_df.columns[0]

    # bar chart (counts)
    fig = px.bar(
        pareto_df,
        x=category_col,
        y="count",
        labels={category_col: "Category", "count": "Count"},
    )

    # cumulative percentage line
    fig.add_trace(
        go.Scatter(
            x=pareto_df[category_col],
            y=pareto_df["cum_pct"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2"
        )
    )

    # two y-axes: left = counts, right = %
    fig.update_layout(
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100]  # keep percentage scale
        ),
        title="Pareto Chart"
    )

    return fig
