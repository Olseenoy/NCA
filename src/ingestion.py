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

def confirm_switch_input(current_method, new_method):
    """
    Checks if switching input method while data exists.
    Returns True to proceed, False to cancel.
    """
    if "df" in st.session_state and st.session_state.df is not None:
        proceed = st.warning(
            f"You are switching from **{current_method}** to **{new_method}**. "
            "This will terminate ongoing analysis.", icon="⚠️"
        )
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Continue Switching"):
                st.session_state.df = None  # Clear current data
                return True
        with cols[1]:
            if st.button("Cancel Switching"):
                st.experimental_rerun()  # Refresh page without switching
                return False
        return False
    return True

def manual_log_entry():
    """
    Allows manual entry of up to 5 logs with up to 10 fields each via Streamlit.
    Uses session state for navigation & auto-fills field names from Log 1.
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
            st.rerun()
    with col_next:
        if current_log < num_logs and st.button("Next Log"):
            st.session_state.current_log += 1
            st.rerun()

    # Finalize entry
    if current_log == num_logs and st.button("Save Manual Logs"):
        df = pd.DataFrame(st.session_state.logs)
        for col in df.columns:
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

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    df.to_parquet(path, engine='pyarrow')
    return path
