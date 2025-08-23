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
    Manual entry of up to 5 logs with up to 10 fields.
    Side-by-side Field/Content, auto-fill from Log 1, session state navigation.
    """
    st.write("### Manual Log Entry")

    # Initialize session state
    if "num_logs" not in st.session_state:
        st.session_state.num_logs = 1
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1
    if "logs" not in st.session_state:
        st.session_state.logs = [{}]

    # Number of logs input
    num_logs = st.number_input(
        "Number of Logs",
        min_value=1,
        max_value=5,
        value=st.session_state.num_logs,
        step=1
    )

    # Only update logs list if number of logs changed
    if num_logs != st.session_state.num_logs:
        st.session_state.num_logs = num_logs
        old_logs = st.session_state.logs
        st.session_state.logs = old_logs[:num_logs] + [{} for _ in range(num_logs - len(old_logs))]

    current_log = st.session_state.current_log
    st.subheader(f"Log {current_log} of {st.session_state.num_logs}")

    # Field names from Log 1 template
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
            default_value = st.session_state.logs[current_log - 1].get(field, "")
            value = st.text_input(
                f"Content {i}",
                value=default_value,
                key=f"value_{current_log}_{i}"
            )

        if field:
            entry[field] = value
            # Auto-fill remaining logs
            for j in range(current_log, st.session_state.num_logs):
                st.session_state.logs[j][field] = value

    # Save current log
    st.session_state.logs[current_log - 1] = entry

    # Navigation buttons
    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_log > 1 and st.button("Previous Log"):
            st.session_state.current_log -= 1
            st.experimental_rerun()
    with col_next:
        if current_log < st.session_state.num_logs and st.button("Next Log"):
            st.session_state.current_log += 1
            st.experimental_rerun()

    # Save all logs button
    if st.button("Save All Logs"):
        df = pd.DataFrame(st.session_state.logs)
        for col in df.columns:
            df[col] = df[col].astype(str)
        df.index = df.index + 1  # start index from 1
        st.write("### Raw Data Preview")
        st.dataframe(df)
        return df

    return None


def save_processed(df: pd.DataFrame, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    df.to_parquet(path, engine='pyarrow')
    return path
