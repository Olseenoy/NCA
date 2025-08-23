import pandas as pd
from config import PROCESSED_DIR
import streamlit as st

def ingest_file(file_obj):
    """Reads CSV or Excel from Streamlit uploader or local path."""
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
    Allows manual entry of up to 5 logs with up to 10 fields each via Streamlit.
    Returns DataFrame once all logs are entered.
    """
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=5, value=1)

    all_entries = []

    for log_num in range(1, num_logs + 1):
        st.subheader(f"Log {log_num}")
        entry = {}
        for i in range(1, 11):  # up to 10 fields
            field = st.text_input(f"Field {i} Name (Log {log_num})", key=f"field_{log_num}_{i}")
            value = st.text_input(f"Content {i} (Log {log_num})", key=f"value_{log_num}_{i}")
            if field:
                entry[field] = value
        all_entries.append(entry)

    if st.button("Save Manual Logs"):
        df = pd.DataFrame(all_entries)
        # Convert all object columns to string to avoid parquet errors
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        st.write("### Raw Data Preview", df)
        return df
    return None

def save_processed(df: pd.DataFrame, name: str):
    """
    Saves DataFrame as a Parquet file after ensuring object columns are stringified.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name

    # Ensure all object columns are strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Save to Parquet
    df.to_parquet(path, engine='pyarrow')
    return path
