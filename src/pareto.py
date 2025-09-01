# src/pareto.py
import pandas as pd
from typing import Optional

def pareto_table(df: pd.DataFrame, category_col: str, weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Generates a Pareto table with counts and cumulative percentage.
    """
    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in DataFrame.")
    if weight_col and weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame.")
    if df.empty:
        return pd.DataFrame(columns=[category_col, 'count', 'cum_pct'])

    if weight_col:
        s = df.groupby(category_col)[weight_col].sum().sort_values(ascending=False)
    else:
        s = df[category_col].value_counts()

    cum = s.cumsum() / s.sum()
    tab = pd.DataFrame({'count': s, 'cum_pct': cum})
    tab = tab.reset_index().rename(columns={'index': category_col})
    return tab
