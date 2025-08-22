import pandas as pd
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR

def ingest_file(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    return df

def save_processed(df: pd.DataFrame, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path)
    return path
