# src/ingestion.py

import os
import io
import json
from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st
import requests

# Import your project config where PROCESSED_DIR is defined
try:
    from config import PROCESSED_DIR
except Exception:
    PROCESSED_DIR = os.path.join(os.getcwd(), "processed")

# Ensure directory path is consistent
PROCESSED_DIR = PROCESSED_DIR if PROCESSED_DIR else "data/processed"

# -----------------------------------------
# Safe rerun helper for Streamlit versions
# -----------------------------------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------------------
# Reset session state when switching input
# -----------------------------------------
def reset_session():
    """
    Clears all Streamlit session state and reruns the app.
    Useful when switching between different data input types.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    safe_rerun()

def select_input_type():
    input_type = st.selectbox(
        "Select Data Input Type",
        ["Manual Entry", "File Upload", "Google Sheet", "OneDrive", "REST API", "Database", "MongoDB"]
    )

    # Detect change and reset session
    if "prev_input_type" not in st.session_state:
        st.session_state.prev_input_type = input_type

    if input_type != st.session_state.prev_input_type:
        reset_session()

    st.session_state.prev_input_type = input_type
    return input_type

# -----------------------------------------
# Local file ingestion (CSV + Excel)
# -----------------------------------------
def ingest_file(file_obj) -> pd.DataFrame:
    if hasattr(file_obj, "name"):
        filename = file_obj.name.lower()
    else:
        filename = str(file_obj).lower()

    if filename.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
        except Exception:
            raise ImportError("openpyxl is required to read Excel files. Install with `pip install openpyxl`.")
        try:
            return pd.read_excel(file_obj, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file: {e}")
    elif filename.endswith(".csv"):
        try:
            return pd.read_csv(file_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")
    else:
        raise ValueError("Unsupported file format. Only CSV, XLSX, and XLS are supported.")

# -----------------------------------------
# Google Sheets ingestion
# -----------------------------------------
def ingest_google_sheet(
    url_or_id: str,
    service_account_json_path: Optional[str] = None,
    api_key: Optional[str] = None,
    worksheet_index: int = 0,
) -> pd.DataFrame:
    # Service account mode
    if service_account_json_path and os.path.exists(service_account_json_path):
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
        except Exception as e:
            raise ImportError(
                "gspread and oauth2client are required for service-account Google Sheets access. "
                "Install: pip install gspread oauth2client"
            ) from e

        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_path, scope)
        client = gspread.authorize(creds)

        try:
            if "docs.google.com" in url_or_id:
                sh = client.open_by_url(url_or_id)
            else:
                sh = client.open_by_key(url_or_id)
            ws = sh.get_worksheet(worksheet_index)
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            raise RuntimeError(f"Failed to read Google Sheet with service account: {e}") from e

    # Public CSV export
    if url_or_id.startswith("http") and "export?format=csv" in url_or_id:
        try:
            params = {"key": api_key} if api_key else None
            r = requests.get(url_or_id, params=params, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Google Sheet CSV export URL: {e}") from e

    # Attempt to extract sheet id
    sheet_id = url_or_id
    if "/" in url_or_id:
        try:
            parts = url_or_id.split("/")
            if "d" in parts:
                idx = parts.index("d")
                sheet_id = parts[idx + 1]
        except Exception:
            pass

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
    try:
        r = requests.get(export_url, params={"key": api_key} if api_key else None, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Google Sheet via CSV export: {e}") from e

# -----------------------------------------
# OneDrive / Microsoft Graph ingestion
# -----------------------------------------
def ingest_onedrive(file_path_or_share_url: str, access_token: Optional[str] = None,
                    client_id: Optional[str] = None, client_secret: Optional[str] = None,
                    tenant_id: Optional[str] = None) -> pd.DataFrame:
    token = access_token
    if not token:
        if not (client_id and client_secret and tenant_id):
            raise EnvironmentError("Provide access_token or client_id+client_secret+tenant_id for OneDrive ingestion.")
        try:
            from msal import ConfidentialClientApplication
        except Exception:
            raise ImportError("msal is required for Microsoft Graph authentication. Install with `pip install msal`.")
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        app = ConfidentialClientApplication(client_id, authority=authority, client_credential=client_secret)
        token_resp = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if not token_resp or "access_token" not in token_resp:
            raise RuntimeError(f"Failed to acquire OneDrive access token: {token_resp}")
        token = token_resp["access_token"]

    headers = {"Authorization": f"Bearer {token}"}

    if file_path_or_share_url.startswith("http"):
        try:
            r = requests.get(file_path_or_share_url, headers=headers, timeout=30)
            if r.status_code == 200 and r.content:
                return _df_from_bytes(r.content, file_path_or_share_url)
        except Exception:
            pass

    if not file_path_or_share_url.startswith("/"):
        path = f"/drive/root:/{file_path_or_share_url}:/content"
    else:
        path = f"/drive{file_path_or_share_url}:/content"

    graph_url = f"https://graph.microsoft.com/v1.0{path}"
    try:
        r = requests.get(graph_url, headers=headers, timeout=30)
        r.raise_for_status()
        return _df_from_bytes(r.content, file_path_or_share_url)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from OneDrive/Graph API: {e}") from e

def _df_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    try:
        import openpyxl
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception:
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(f"Unable to parse downloaded file {filename} as Excel or CSV: {e}") from e

# -----------------------------------------
# REST API ingestion
# -----------------------------------------
def ingest_rest_api(url: str, method: str = "GET", params: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None, json_body: Optional[Dict[str, Any]] = None,
                    auth: Optional[Any] = None, timeout: int = 30) -> pd.DataFrame:
    try:
        r = requests.request(method, url, params=params or {}, headers=headers or {},
                             json=json_body, auth=auth, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"REST API request failed: {e}") from e

    content_type = r.headers.get("content-type", "")
    text = r.text.strip()

    if "application/json" in content_type or text.startswith("[") or text.startswith("{"):
        try:
            data = r.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = [data]
        return pd.DataFrame(data)

    try:
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise ValueError(f"Failed to parse API response as CSV: {e}") from e

# -----------------------------------------
# SQL Database ingestion
# -----------------------------------------
def ingest_database(connection_string: str, query: str) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine
    except Exception:
        raise ImportError("sqlalchemy is required for database ingestion. Install with `pip install sqlalchemy`.")

    try:
        engine = create_engine(connection_string)
    except Exception as e:
        raise RuntimeError(f"Failed to create SQLAlchemy engine: {e}") from e

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        raise RuntimeError(f"Failed to execute SQL query: {e}") from e

    return df

# -----------------------------------------
# MongoDB ingestion
# -----------------------------------------
def ingest_mongodb(uri: str, database: str, collection: str, query: Optional[dict] = None) -> pd.DataFrame:
    try:
        from pymongo import MongoClient
    except Exception:
        raise ImportError("pymongo is required for MongoDB ingestion. Install with `pip install pymongo`.")

    try:
        client = MongoClient(uri)
        coll = client[database][collection]
        q = query or {}
        docs = list(coll.find(q))
    except Exception as e:
        raise RuntimeError(f"MongoDB query failed: {e}") from e

    for d in docs:
        if "_id" in d:
            try:
                d["_id"] = str(d["_id"])
            except Exception:
                pass

    return pd.DataFrame(docs)

# -----------------------------------------
# Manual log entry with preview/save
# -----------------------------------------
def manual_log_entry() -> Optional[pd.DataFrame]:
    """
    Manual entry UI that persists each field/value in session_state using stable keys.
    On Save, reconstructs the full DataFrame from session_state so headers and values
    exactly match what the user entered.
    """
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=5, value=1)

    # initialize current_log
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1

    # ensure current_log is within bounds if num_logs changed
    if st.session_state.current_log > num_logs:
        st.session_state.current_log = num_logs

    current_log = st.session_state.current_log
    st.subheader(f"Log {current_log} of {num_logs}")

    # Build UI for current log: 10 field-name / content pairs
    for i in range(1, 11):
        col1, col2 = st.columns([1, 2])
        field_key = f"manual_field_{current_log}_{i}"
        value_key = f"manual_value_{current_log}_{i}"

        # Use existing session_state value as the text_input initial value so it persists
        with col1:
            st.text_input(f"Field {i} Name", value=st.session_state.get(field_key, ""), key=field_key)
        with col2:
            st.text_input(f"Content {i}", value=st.session_state.get(value_key, ""), key=value_key)

    # Navigation (no forced rerun; changing session_state will rerun automatically)
    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_log > 1 and st.button("Previous", key=f"prev_btn_{current_log}"):
            st.session_state.current_log = current_log - 1
    with col_next:
        if current_log < num_logs and st.button("Next", key=f"next_btn_{current_log}"):
            st.session_state.current_log = current_log + 1

    # Save: reconstruct all logs from session_state keys
    if st.button("Save Manual Logs", key="save_manual_logs"):
        rows = []
        # collect rows and the order of first-seen fields (to preserve header order)
        columns_order = []
        for log_idx in range(1, num_logs + 1):
            entry = {}
            for i in range(1, 11):
                fkey = f"manual_field_{log_idx}_{i}"
                vkey = f"manual_value_{log_idx}_{i}"
                fld = (st.session_state.get(fkey) or "").strip()
                val = st.session_state.get(vkey) or ""
                if fld:
                    # store field: value
                    entry[fld] = val
                    # maintain first-seen order
                    if fld not in columns_order:
                        columns_order.append(fld)
            rows.append(entry)

        # If no field names at all, inform user and abort
        if not columns_order:
            st.warning("No field names entered. Please provide at least one field name for the logs.")
            return None

        # Build DataFrame preserving column order
        data_rows = []
        for entry in rows:
            row = {col: entry.get(col, "") for col in columns_order}
            data_rows.append(row)

        df = pd.DataFrame(data_rows, columns=columns_order)

        st.write("### Preview of Entered Logs")
        st.dataframe(df)

        # Persist to session_state and offer saving
        st.session_state["manual_entered_df"] = df
        save_name = st.text_input("Enter filename to save preview (without extension):", value="manual_logs")
        if st.button("Save Preview", key="save_preview_btn"):
            try:
                # call your save_processed function (assumed to be imported)
                save_processed(df, f"{save_name}.parquet")
                st.success(f"Preview saved as {save_name}.parquet")
            except Exception as e:
                st.error(f"Failed to save preview: {e}")

# -----------------------------------------
# Utility: Fix mixed types before saving
# -----------------------------------------
def fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns have consistent types before saving.
    - Converts mixed-type columns to string
    - Leaves clean numeric or datetime columns untouched
    """
    for col in df.columns:
        # Handle object (string / mixed) columns
        if df[col].dtype == object:
            types_in_col = df[col].dropna().apply(type).unique()
            if len(types_in_col) > 1:
                # Mixed types detected, convert everything to string
                df[col] = df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Attempt to coerce to numeric where possible
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -----------------------------------------
# Save processed DataFrame
# -----------------------------------------
def save_processed(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to Parquet. Falls back to CSV if Parquet fails.
    Handles mixed-type columns gracefully.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # Ensure filename ends correctly
    safe_name = filename if filename.endswith(".parquet") else f"{filename}"
    file_path = os.path.join(PROCESSED_DIR, safe_name)

    try:
        # Ensure column types are consistent
        df = fix_mixed_types(df)
        df.to_parquet(file_path, index=False)
    except Exception as e:
        try:
            # Fallback to CSV if Parquet fails
            csv_path = file_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)
            raise RuntimeError(
                f"Parquet write failed ({e}). Saved as CSV to {csv_path}"
            ) from e
        except Exception as e2:
            raise RuntimeError(
                f"Failed to save DataFrame as parquet and CSV: {e2}"
            ) from e2
