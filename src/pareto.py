# src/pareto.py
import pandas as pd
from typing import Optional

def pareto_table(df: pd.DataFrame, category_col: str, weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Generates a Pareto table with counts and cumulative percentage.
    
    Args:
        df: Input DataFrame
        category_col: Column to group by (categorical)
        weight_col: Optional numeric column to sum instead of counting rows
    
    Returns:
        DataFrame with 'count' and 'cum_pct'
    """
    if weight_col:
        s = df.groupby(category_col)[weight_col].sum().sort_values(ascending=False)
    else:
        s = df[category_col].value_counts()
    
    cum = s.cumsum() / s.sum()
    tab = pd.DataFrame({'count': s, 'cum_pct': cum})
    return tab
