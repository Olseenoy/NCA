# src/ingestion.py
import pandas as pd
from config import PROCESSED_DIR
import streamlit as st

def ingest_file(file_obj):
    """
    Reads CSV or Excel from Streamlit uploader or local path.
    """
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

def manual_log_entry():
    """
    Manual data entry (max 5 logs, 10 fields each).
    Returns DataFrame after entry.
    """
    num_logs = st.number_input("Enter number of logs (max 5)", min_value=1, max_value=5, step=1)
    all_logs = []

    for log_index in range(num_logs):
        st.subheader(f"Log {log_index+1} of {num_logs}")
        log_data = {}
        for field_num in range(1, 11):
            col1, col2 = st.columns(2)
            with col1:
                field = st.text_input(f"Data Field {field_num}", key=f"field_{log_index}_{field_num}")
            with col2:
                value = st.text_input(f"Content {field_num}", key=f"value_{log_index}_{field_num}")
            if field:
                log_data[field] = value
        all_logs.append(log_data)

    if st.button("Generate Preview"):
        df = pd.DataFrame(all_logs)
        st.write("### Raw Data Preview", df)
        return df
    return None

def save_processed(df: pd.DataFrame, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path)
    return path
