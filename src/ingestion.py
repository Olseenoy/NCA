# ================================
# File: src/ingestion.py
# ================================
import pandas as pd
from config import PROCESSED_DIR
import streamlit as st
import os
import streamlit as st_version_check  # For version checking

# Optional: Check Streamlit version to warn about compatibility
try:
    if st_version_check.__version__ < "1.10.0":
        st.warning("Streamlit version is older than 1.10.0. Please upgrade to use `st.rerun()` by running `pip install --upgrade streamlit`.")
except Exception:
    pass

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
    Multi-log manual entry using Streamlit session_state.
    Stores logs in session_state; returns None.
    """
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=5, value=1)

    # Initialize session state
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1
    if "logs" not in st.session_state or len(st.session_state.logs) != num_logs:
        # Reset logs to match num_logs
        st.session_state.logs = [{} for _ in range(num_logs)]
        st.session_state.current_log = min(st.session_state.current_log, num_logs)
    if "manual_logs_saved" not in st.session_state:
        st.session_state.manual_logs_saved = False

    current_log = st.session_state.current_log
    if current_log < 1 or current_log > num_logs:
        st.warning(f"Invalid current_log value: {current_log}. Resetting to 1.")
        st.session_state.current_log = 1
        current_log = 1

    st.subheader(f"Log {current_log}")

    # Use first log fields as template only if filled
    field_template = []
    if (
        st.session_state.logs 
        and isinstance(st.session_state.logs[0], dict) 
        and st.session_state.logs[0]
    ):
        field_template = list(st.session_state.logs[0].keys())

    entry = {}
    for i in range(1, 11):
        col1, col2 = st.columns([1, 2])
        with col1:
            default_field = field_template[i-1] if i-1 < len(field_template) else ""
            field = st.text_input(f"Field {i} Name", value=default_field, key=f"field_{current_log}_{i}")
        with col2:
            value = st.text_input(f"Content {i}", key=f"value_{current_log}_{i}")
        if field:
            entry[field] = value

    # Save current log safely
    try:
        if 0 <= current_log - 1 < len(st.session_state.logs):
            st.session_state.logs[current_log - 1] = entry
        else:
            st.error(f"Cannot save log: Invalid index {current_log - 1} for logs list of length {len(st.session_state.logs)}")
            return
    except Exception as e:
        st.error(f"Failed to save log entry: {e}")
        return

    # Navigation buttons
    col_prev, col_next = st.columns(2)
    if col_prev.button("Previous Log") and current_log > 1:
        st.session_state.current_log -= 1
        st.rerun()  # Safe rerun
    if col_next.button("Next Log") and current_log < num_logs:
        st.session_state.current_log += 1
        st.rerun()  # Safe rerun

    # Save logs button (only after last log)
    if current_log == num_logs:
        if st.button("Save Manual Logs"):
            try:
                df = pd.DataFrame(st.session_state.logs)
                for col in df.columns:
                    df[col] = df[col].astype(str)
                st.session_state.manual_logs_saved = True
                st.session_state.manual_logs_df = df
                st.success("Manual logs saved!")
            except Exception as e:
                st.error(f"Failed to save manual logs: {e}")

    return None


def save_processed(df, filename):
    """Save DataFrame to parquet in processed dir."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(file_path, index=False)
