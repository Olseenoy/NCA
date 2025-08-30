# src/pareto.py
import pandas as pd

def pareto_table(
    df: pd.DataFrame,
    category_col: str,
    weight_col: str | None = None,
    top_n: int | None = None
) -> pd.DataFrame:
    """
    Generate Pareto table with counts and cumulative percentage.

    Parameters:
    - df: input DataFrame
    - category_col: column for categories
    - weight_col: optional column to sum instead of counting
    - top_n: optional number of top categories to include

    Returns:
    - DataFrame with columns ['Count', 'Cumulative %']
    """
    if weight_col:
        s = df.groupby(category_col)[weight_col].sum().sort_values(ascending=False)
    else:
        s = df[category_col].value_counts()

    if top_n:
        s = s.head(top_n)

    cum_pct = s.cumsum() / s.sum() * 100  # percentage 0-100
    tab = pd.DataFrame({
        'Count': s,
        'Cumulative %': cum_pct
    })
    return tab

