# src/ingestion.py
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

def ingest_file(file_obj):
    """
    Reads CSV or Excel from Streamlit uploader or local path.
    """
    # If file_obj is a Streamlit UploadedFile, use its name for format detection
    if hasattr(file_obj, "name"):
        filename = file_obj.name.lower()
    else:
        filename = str(file_obj).lower()

    if filename.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Please install openpyxl: pip install openpyxl")
        return pd.read_excel(file_obj, engine="openpyxl")
    else:
        return pd.read_csv(file_obj)

def save_processed(df: pd.DataFrame, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path)
    return path
