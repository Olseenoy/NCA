# src/preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    tokens = [t for t in s.split() if t not in STOP]
    return ' '.join(tokens)


def preprocess_df(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    df = df.copy()
    # ensure all selected columns are strings
    df[text_cols] = df[text_cols].astype(str)
    # create single text field
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    df['clean_text'] = df['combined_text'].apply(clean_text)
    return df
