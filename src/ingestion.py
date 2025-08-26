import os
import pandas as pd
import streamlit as st
from config import PROCESSED_DIR
from sqlalchemy import create_engine
import requests
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# -----------------------------------------
# Utility to rerun Streamlit
# -----------------------------------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------------------
# Local file ingestion
# -----------------------------------------
def ingest_file(file_obj):
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

# -----------------------------------------
# Google Sheets ingestion
# -----------------------------------------
def ingest_google_sheet(sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not creds_path or not os.path.exists(creds_path):
        st.error("Google service account JSON not found in environment.")
        return None

    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_url(sheet_url).sheet1
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# -----------------------------------------
# Database ingestion
# -----------------------------------------
def ingest_database(connection_string, query):
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

# -----------------------------------------
# REST API ingestion (ERP/MES/QMS)
# -----------------------------------------
def ingest_api(endpoint_url, headers=None):
    response = requests.get(endpoint_url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"API Request failed with status: {response.status_code}")
        return None

# -----------------------------------------
# Manual log entry
# -----------------------------------------
def manual_log_entry():
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=5, value=1)

    if "current_log" not in st.session_state:
        st.session_state.current_log = 1
    if "logs" not in st.session_state or len(st.session_state.logs) != num_logs:
        st.session_state.logs = [{} for _ in range(num_logs)]

    current_log = st.session_state.current_log
    st.subheader(f"Log {current_log}")

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

    st.session_state.logs[current_log - 1] = entry

    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_log > 1 and st.button("Previous Log"):
            st.session_state.current_log -= 1
            safe_rerun()
    with col_next:
        if current_log < num_logs and st.button("Next Log"):
            st.session_state.current_log += 1
            safe_rerun()

    if current_log == num_logs and st.button("Save Manual Logs"):
        df = pd.DataFrame(st.session_state.logs)
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
    return None

# -----------------------------------------
# Save DataFrame
# -----------------------------------------
def save_processed(df, filename):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(file_path, index=False)
