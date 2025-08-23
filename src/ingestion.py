import pandas as pd
from config import PROCESSED_DIR
import streamlit as st
import os

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
    Uses session state for navigation & auto-fills field names from Log 1.
    Returns final DataFrame after save, otherwise None.
    """
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=5, value=1)

    # Initialize session state
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1
    if "logs" not in st.session_state or len(st.session_state.logs) != num_logs:
        st.session_state.logs = [{} for _ in range(num_logs)]

    current_log = st.session_state.current_log
    st.subheader(f"Log {current_log}")

    # Use Log 1 field names as template
    field_template = list(st.session_state.logs[0].keys()) if current_log > 1 else []

    entry = {}
    for i in range(1, 11):
        col1, col2 = st.columns([1, 2])
        with col1:
            default_field = field_template[i-1] if i-1 < len(field_template) else ""
            field = st.text_input(
                f"Field {i} Name",
                value=default_field,
                key=f"field_{current_log}_{i}"
            )
        with col2:
            value = st.text_input(
                f"Content {i}",
                key=f"value_{current_log}_{i}"
            )
        if field:
            entry[field] = value

    # Save current log data to session state
    st.session_state.logs[current_log - 1] = entry

    # Navigation buttons
    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_log > 1 and st.button("Previous Log"):
            st.session_state.current_log -= 1
            st.experimental_rerun()
    with col_next:
        if current_log < num_logs and st.button("Next Log"):
            st.session_state.current_log += 1
            st.experimental_rerun()

    # Finalize entry (only returns DataFrame, no preview here)
    if current_log == num_logs and st.button("Save Manual Logs"):
        df = pd.DataFrame(st.session_state.logs)
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
    return None


def save_processed(df, filename):
    """Save DataFrame to parquet in processed dir."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(file_path, index=False)

