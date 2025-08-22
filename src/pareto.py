# src/pareto.py
import pandas as pd

def pareto_table(df: pd.DataFrame, category_col: str, weight_col: str|None=None):
    if weight_col:
        s = df.groupby(category_col)[weight_col].sum().sort_values(ascending=False)
    else:
        s = df[category_col].value_counts()
    cum = s.cumsum() / s.sum()
    tab = pd.DataFrame({'count': s, 'cum_pct': cum})
    return tab
