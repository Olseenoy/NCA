# src/ingestion.py
"""
ingestion.py

Multi-source ingestion utilities for Smart NC Analyzer.

Supported sources:
- Local CSV / Excel (explicit CSV handling)
- Google Sheets:
    * Service-account JSON (preferred)
    * Public CSV export (via API key or direct export URL)
- OneDrive / SharePoint via Microsoft Graph:
    * Accepts either an access token (short-term) or performs client_credentials flow using MSAL
- Generic REST API (JSON array or CSV)
- SQL Databases via SQLAlchemy
- MongoDB via pymongo
- Manual log entry (Streamlit UI helper)
- Save processed DataFrame to PROCESSED_DIR (config.PROCESSED_DIR)

Design notes:
- All cloud/database ingestion functions accept credential arguments (so the Streamlit UI may provide creds
  from .env or user input).
- Optional dependencies (gspread, msal, pymongo, sqlalchemy, openpyxl) are imported lazily with clear error messages.
"""

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

# -----------------------------------------
# Safe rerun helper for Streamlit versions
# -----------------------------------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------------------
# Local file ingestion (explicit CSV + Excel)
# -----------------------------------------
def ingest_file(file_obj) -> pd.DataFrame:
    """
    Read CSV or Excel from Streamlit uploader or local path.

    - If file_obj has .name (Streamlit upload), uses that filename to infer type.
    - Accepts a file-like object or a file path string.
    Raises ValueError for unsupported formats.
    """
    if hasattr(file_obj, "name"):
        filename = file_obj.name.lower()
    else:
        filename = str(file_obj).lower()

    # Explicit checks for clarity
    if filename.endswith((".xlsx", ".xls")):
        try:
            import openpyxl  # ensure dependency present
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
# Google Sheets ingestion (service-account or CSV export)
# -----------------------------------------
def ingest_google_sheet(
    url_or_id: str,
    service_account_json_path: Optional[str] = None,
    api_key: Optional[str] = None,
    worksheet_index: int = 0,
) -> pd.DataFrame:
    """
    Ingest Google Sheet.

    Modes:
    1. Service account JSON (recommended): provide service_account_json_path (path to JSON file)
       -> uses gspread + oauth2client to read sheet.
    2. CSV export: provide api_key (optional) and sheet id or export URL. This builds the export URL:
       https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}

    Parameters:
        url_or_id: either full sheet URL, sheet id, or direct export URL (with export?format=csv)
        service_account_json_path: optional path to JSON credentials
        api_key: optional API key for public sheets
        worksheet_index: worksheet index (0-based) when using service account mode

    Returns pandas.DataFrame or raises informative exceptions.
    """
    # Mode 1: service account
    if service_account_json_path and os.path.exists(service_account_json_path):
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
        except Exception as e:
            raise ImportError("gspread and oauth2client are required for service-account Google Sheets access. Install: pip install gspread oauth2client") from e

        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_path, scope)
        client = gspread.authorize(creds)

        # Accept URL or spreadsheet id
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

    # Mode 2: CSV export (public or published sheets)
    # If user provided full export URL (contains export?format=csv), use it directly
    if url_or_id.startswith("http") and "export?format=csv" in url_or_id:
        try:
            params = {"key": api_key} if api_key else None
            r = requests.get(url_or_id, params=params, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Google Sheet CSV export URL: {e}") from e

    # Try to extract sheet id and build export url with gid=0
    sheet_id = url_or_id
    if "/" in url_or_id:
        # crude attempt to extract id
        try:
            parts = url_or_id.split("/")
            if "d" in parts:
                idx = parts.index("d")
                sheet_id = parts[idx + 1]
            else:
                # fallback: look for /spreadsheets/d/<id> pattern
                for i, p in enumerate(parts):
                    if p == "d" and i + 1 < len(parts):
                        sheet_id = parts[i + 1]
                        break
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
def ingest_onedrive(
    file_path_or_share_url: str,
    access_token: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download file from OneDrive/SharePoint via Microsoft Graph and parse as DataFrame.

    Two modes:
    - If access_token provided, use it directly (useful for short-term tokens).
    - Else, attempt client_credentials flow using client_id, client_secret, tenant_id via msal.

    file_path_or_share_url can be:
    - A direct sharing URL to the file (public link)
    - A Graph path like '/drive/root:/path/to/file.xlsx:/content' (we will try constructing)
    """
    # Acquire token if not provided
    token = access_token
    if not token:
        # require client credentials
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

    # If it's a sharing URL, try to fetch directly
    if file_path_or_share_url.startswith("http"):
        # Try a direct GET first; some sharing URLs will redirect or return content
        try:
            r = requests.get(file_path_or_share_url, headers=headers, timeout=30)
            if r.status_code == 200 and r.content:
                return _df_from_bytes(r.content, file_path_or_share_url)
        except Exception:
            # fall through to Graph API approach
            pass

    # Fallback: try Graph API path construction
    # If user supplied a path like '/path/to/file.xlsx', convert to drive/root:/...:/content
    if not file_path_or_share_url.startswith("/"):
        path = f"/drive/root:/{file_path_or_share_url}:/content"
    else:
        # assume user already gave a drive path (e.g., /drive/root:/path/to/file.xlsx:/content)
        path = f"/drive{file_path_or_share_url}:/content"

    graph_url = f"https://graph.microsoft.com/v1.0{path}"
    try:
        r = requests.get(graph_url, headers=headers, timeout=30)
        r.raise_for_status()
        return _df_from_bytes(r.content, file_path_or_share_url)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from OneDrive/Graph API: {e}") from e

def _df_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """
    Try to parse downloaded bytes as Excel first, then CSV.
    """
    # Try Excel first
    try:
        import openpyxl  # noqa: F401
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception:
        # Try CSV decode
        try:
            text = content.decode("utf-8")
        except Exception:
            # fallback: try latin-1
            text = content.decode("latin-1", errors="ignore")
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(f"Unable to parse downloaded file {filename} as Excel or CSV: {e}") from e

# -----------------------------------------
# Generic REST API ingestion
# -----------------------------------------
def ingest_rest_api(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    auth: Optional[Any] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch data from REST API and return DataFrame.

    Accepts JSON arrays of objects or CSV responses. It will try to detect content-type.
    """
    try:
        r = requests.request(method, url, params=params or {}, headers=headers or {}, json=json_body, auth=auth, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"REST API request failed: {e}") from e

    content_type = r.headers.get("content-type", "")
    text = r.text.strip()

    # JSON
    if "application/json" in content_type or text.startswith("[") or text.startswith("{"):
        try:
            data = r.json()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e

        # If dict, try to find list inside
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                # wrap single dict as list
                data = [data]
        return pd.DataFrame(data)

    # CSV / text
    try:
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise ValueError(f"Failed to parse API response as CSV: {e}") from e

# -----------------------------------------
# SQL Database ingestion using SQLAlchemy
# -----------------------------------------
def ingest_database(connection_string: str, query: str) -> pd.DataFrame:
    """
    Ingest from SQL Database via SQLAlchemy.

    Example connection strings:
    - postgresql+psycopg2://user:pass@host:port/dbname
    - mysql+pymysql://user:pass@host:port/dbname
    - mssql+pyodbc://user:pass@dsn
    - sqlite:///./mydb.sqlite
    """
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
# MongoDB ingestion using pymongo
# -----------------------------------------
def ingest_mongodb(uri: str, database: str, collection: str, query: Optional[dict] = None) -> pd.DataFrame:
    """
    Ingest documents from MongoDB collection into DataFrame.
    """
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

    # Convert ObjectId to string and return DataFrame
    for d in docs:
        if "_id" in d:
            try:
                d["_id"] = str(d["_id"])
            except Exception:
                pass

    return pd.DataFrame(docs)

# -----------------------------------------
# Manual log entry (Streamlit UI helper)
# -----------------------------------------
def manual_log_entry() -> Optional[pd.DataFrame]:
    """
    Allows manual entry of up to 20 logs with up to 10 fields each via Streamlit.
    Uses session state for navigation & auto-fills field names from Log 1.
    Returns final DataFrame after Save (or None while editing).
    """
    st.write("### Manual Log Entry")
    num_logs = st.number_input("Number of Logs", min_value=1, max_value=20, value=1)

    # Initialize session state safely
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1
    if "logs" not in st.session_state or len(st.session_state.get("logs", [])) != num_logs:
        st.session_state.logs = [{} for _ in range(num_logs)]

    current_log = st.session_state.current_log
    st.subheader(f"Log {current_log}")

    # Use Log 1 field names as template
    field_template = list(st.session_state.logs[0].keys()) if st.session_state.logs and current_log > 1 else []

    entry = {}
    for i in range(1, 11):
        col1, col2 = st.columns([1, 2])
        with col1:
            default_field = field_template[i - 1] if i - 1 < len(field_template) else f"Field_{i}"
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
            safe_rerun()
    with col_next:
        if current_log < num_logs and st.button("Next Log"):
            st.session_state.current_log += 1
            safe_rerun()

    # Finalize entry (only returns DataFrame after user confirms)
    if current_log == num_logs and st.button("Save Manual Logs"):
        df = pd.DataFrame(st.session_state.logs)
        for col in df.columns:
            df[col] = df[col].astype(str)
        # reset navigation so next invocation starts fresh
        st.session_state.current_log = 1
        st.session_state.logs = []
        return df

    return None


# -----------------------------------------
# Save processed DataFrame to PROCESSED_DIR
# -----------------------------------------
def save_processed(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to parquet in PROCESSED_DIR.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # sanitize filename
    safe_name = filename if filename.endswith(".parquet") else f"{filename}"
    file_path = os.path.join(PROCESSED_DIR, safe_name)
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        # fallback to csv if parquet fails
        try:
            csv_path = file_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)
            raise RuntimeError(f"Parquet write failed ({e}). Saved as CSV to {csv_path}") from e
        except Exception as e2:
            raise RuntimeError(f"Failed to save DataFrame as parquet and CSV: {e2}") from e2
