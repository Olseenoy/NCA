# src/ingestion.py
import pandas as pd
from pathlib import Path
from .config import RAW_DIR, PROCESSED_DIR


def ingest_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def save_processed(df: pd.DataFrame, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path)
    return path
