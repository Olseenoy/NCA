# ================================
# File: src/streamlit_app.py
# ================================
import os
import re
import nltk
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from nltk.stem import WordNetLemmatizer        
from fuzzywuzzy import process
from reportlab.platypus import Image as RLImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image as PILImage 
from typing import Optional, Dict
from dotenv import load_dotenv, set_key, find_dotenv
from io import BytesIO
from collections import Counter   
from collections import defaultdict
# Local imports (same src/ folder)
# from rca_engine import process_uploaded_docs, extract_recurring_issues, ai_rca_with_fallback
from rca_engine import ai_rca_with_fallback
from visualization import rule_based_rca_fallback, visualize_fishbone_plotly




# --------------------------
# Fishbone Helpers
# --------------------------
import re
import plotly.graph_objects as go

import re

def extract_main_points(result, raw_text: str = ""):
    """
    Extracts ONLY points from 'Possible Root Causes' and cleans them.
    Example: '1. ** Wear and Tear' -> 'Wear and Tear'
    """
    points = []

    # --- 1. If structured list exists, use it ---
    if result.get("possible_root_causes"):
        for p in result["possible_root_causes"]:
            clean = re.sub(r"^\d+[\.\)]\s*", "", p)  # remove "1." or "2)"
            clean = clean.strip("-•* ").strip()
            clean = re.sub(r"\*+", "", clean)        # remove extra asterisks
            if ":" in clean:
                clean = clean.split(":", 1)[0].strip()
            if clean:
                points.append(f"- {clean}")   # 👈 add dash here
        return points

    # --- 2. Extract from raw_text section ---
    if raw_text:
        lines = raw_text.splitlines()
        capture = False
        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            if "possible root cause" in clean.lower():
                capture = True
                continue

            if capture and any(h in clean.lower() for h in ["capa", "corrective", "preventive", "action plan"]):
                break

            if capture:
                clean = re.sub(r"^\d+[\.\)]\s*", "", clean)
                clean = clean.strip("-•* ").strip()
                clean = re.sub(r"\*+", "", clean)
                if ":" in clean:
                    clean = clean.split(":", 1)[0].strip()
                if clean:
                    points.append(clean)

    return points




def categorize_6m(points):
    """
    Categorize extracted points into 6M buckets.
    Uses keyword matching (basic heuristic).
    """
    categories = {
        "Man": [],
        "Machine": [],
        "Method": [],
        "Material": [],
        "Measurement": [],
        "Environment": []
    }
    
    for p in points:
        low = p.lower()
        if any(k in low for k in ["operator", "training", "staff", "human", "worker"]):
            categories["Man"].append(p)
        elif any(k in low for k in ["machine", "equipment", "motor", "gear", "seal"]):
            categories["Machine"].append(p)
        elif any(k in low for k in ["process", "procedure", "method", "technique"]):
            categories["Method"].append(p)
        elif any(k in low for k in ["material", "raw", "supply", "milk", "seal"]):
            categories["Material"].append(p)
        elif any(k in low for k in ["measure", "test", "inspection", "calibration"]):
            categories["Measurement"].append(p)
        elif any(k in low for k in ["environment", "humidity", "temperature", "clean"]):
            categories["Environment"].append(p)
        else:
            categories["Method"].append(p)  # default bucket
    
    return categories


import textwrap
import plotly.graph_objects as go

def visualize_fishbone_plotly(categories, wrap_width=25):
    """
    Fishbone diagram (Ishikawa).
    Category placed at branch edge, causes listed under it (right aligned, compact size).
    Ensures top/bottom labels are never cut off.
    """
    fig = go.Figure()

    # Count max number of wrapped lines (for spacing)
    max_lines = 0
    for causes in categories.values():
        for c in causes:
            wrapped = textwrap.wrap(c, width=wrap_width)
            max_lines = max(max_lines, len(wrapped))

    # Vertical offset for main spine
    y_offset = -0.3 * max_lines  

    # Main spine
    fig.add_trace(go.Scatter(
        x=[0, 10], y=[0+y_offset, 0+y_offset],
        mode="lines", line=dict(color="black", width=3),
        showlegend=False
    ))

    # Branch positions
    branches = {
        "Man": (2, 1),
        "Machine": (4, 1),
        "Method": (6, 1),
        "Material": (8, 1),
        "Measurement": (3, -1),
        "Environment": (7, -1)
    }

    for cat, (x, y) in branches.items():
        # Branch line
        fig.add_trace(go.Scatter(
            x=[x, x+1], y=[0+y_offset, y+y_offset],
            mode="lines", line=dict(color="black", width=2),
            showlegend=False
        ))

        # Wrap causes
        causes_wrapped = []
        for c in categories.get(cat, []):
            wrapped = "<br>".join(textwrap.wrap(c, width=wrap_width))
            causes_wrapped.append(f"- {wrapped}")

        # Category + causes text
        if causes_wrapped:
            text_label = f"<b>{cat}</b><br>{'<br>'.join(causes_wrapped)}"
        else:
            text_label = f"<b>{cat}</b>"

        # Add text
        fig.add_trace(go.Scatter(
            x=[x+1.05], y=[y+y_offset - 0.1 if y > 0 else y+y_offset + 0.1],
            text=[text_label],
            mode="text",
            textposition="top right" if y > 0 else "bottom right",
            textfont=dict(family="Arial", size=10),
            showlegend=False
        ))

    # Expand y-axis so nothing is clipped
    fig.update_layout(
        title="Fishbone Diagram (Ishikawa)",
        xaxis=dict(visible=False, range=[0, 11]),
        yaxis=dict(visible=False, range=[-3+ y_offset, 3+ y_offset]),  # extra padding
        plot_bgcolor="white",
        height=550,
        margin=dict(l=40, r=40, t=80, b=60)
    )

    return fig


# --- Markdown → PDF flowable converter ---

import re
from reportlab.platypus import (
    Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

def convert_markdown_to_pdf_content(raw_text, styles):
    flowables = []
    lines = raw_text.splitlines()

    # --- Custom Normal style ---
    normal_style = ParagraphStyle(
        "Justify12",
        parent=styles["Normal"],
        fontSize=12,
        leading=14,   # ~1.0 line spacing
        alignment=TA_JUSTIFY
    )

    bullet_buffer = []
    in_action_plan = False
    action_rows = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            flowables.append(
                ListFlowable(
                    [ListItem(Paragraph(item, normal_style)) for item in bullet_buffer],
                    bulletType='1' if bullet_buffer[0][0].isdigit() else 'bullet'
                )
            )
            bullet_buffer = []
            flowables.append(Spacer(1, 10))

    def md_to_html(text: str) -> str:
        """Convert markdown (bold only) to safe HTML for ReportLab."""
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    for line in lines:
        line = line.strip()
        if not line:
            flush_bullets()
            continue

        # Detect Action Plan heading
        if line.lower().startswith("**action plan"):
            flush_bullets()
            in_action_plan = True
            flowables.append(Paragraph("<b>Action Plan</b>", styles["Heading2"]))
            flowables.append(Spacer(1, 6))
            action_rows = [[
                Paragraph("<b>Action</b>", normal_style),
                Paragraph("<b>Owner</b>", normal_style),
                Paragraph("<b>Timeline</b>", normal_style),
                Paragraph("<b>Status</b>", normal_style),
            ]]
            continue

        if in_action_plan and (line.startswith("-") or line[0].isdigit()):
            action = line.lstrip("•-0123456789. ").strip()
            action_rows.append([
                Paragraph(md_to_html(action), normal_style),
                Paragraph(" ", normal_style),
                Paragraph(" ", normal_style),
                Paragraph(" ", normal_style),
            ])
            continue

        # Headings
        if line.startswith("**") and line.endswith("**"):
            flush_bullets()
            flowables.append(Paragraph(md_to_html(line), styles["Heading2"]))
            flowables.append(Spacer(1, 6))

        # Numbered / bulleted list
        elif line.startswith("•") or line[0].isdigit() or line.startswith("-"):
            # Remove bullets AND numbers like "1.", "2." at the start
            clean_line = re.sub(r"^[\d\.\-\•\s]+", "", line).strip()
            bullet_buffer.append(md_to_html(clean_line))


        # Normal paragraph
        else:
            flush_bullets()
            flowables.append(Paragraph(md_to_html(line), normal_style))
            flowables.append(Spacer(1, 6))

    flush_bullets()

    # Action Plan table
    if action_rows:
        table = Table(action_rows, colWidths=[250, 80, 80, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 12))

    return flowables








# -----------------------------
# Ensure import paths are correct
# -----------------------------
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
for p in [FILE_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Local package imports (ingestion functions expected to be present)
from ingestion import (
    ingest_file,
    ingest_google_sheet,
    ingest_onedrive,
    ingest_rest_api,
    ingest_database,
    ingest_mongodb,
    manual_log_entry,
    save_processed,
)

from preprocessing import preprocess_df
from embeddings import embed_texts
from clustering import fit_kmeans, evaluate_kmeans
from visualization import (
    pareto_plot,
    cluster_scatter,
    plot_spc_chart,
    plot_trend_dashboard,
    plot_time_series_trend,
)
from pareto import pareto_table
from db import init_db, SessionLocal, CAPA
# from rca_engine import process_uploaded_docs, extract_recurring_issues, ai_rca_with_fallback
from fishbone_visualizer import visualize_fishbone


# Load .env if present
load_dotenv()

# ---------------- Safe Rerun ----------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ----------------- Helpers -----------------
def _make_unique(names):
    seen = {}
    unique = []
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"col_{i+1}"
        base = n
        cnt = seen.get(base, 0)
        if cnt:
            n = f"{base}_{cnt+1}"
        seen[base] = cnt + 1
        unique.append(n)
    return unique


def apply_row_as_header(raw_df: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df

    row_idx = int(max(0, min(row_idx, len(raw_df) - 1)))

    new_header = raw_df.iloc[row_idx].astype(str).tolist()
    new_header = _make_unique(new_header)

    # Drop all rows up to and including the header row
    df = raw_df.drop(index=range(0, row_idx + 1)).copy()
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    for col in df.columns:
        if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
                except Exception:
                    pass


    # ✅ Ensure MACHINE NO is always treated as string
    if "MACHINE NO" in df.columns:
        df["MACHINE NO"] = df["MACHINE NO"].astype(str)

    return df



# ----------------- Credentials helpers -----------------
CRED_KEYS = {
    "GOOGLE_SERVICE_ACCOUNT_JSON": "Path to Google service account JSON",
    "GOOGLE_API_KEY": "Google Sheets API Key (optional)",
    "ONEDRIVE_CLIENT_ID": "OneDrive Client ID",
    "ONEDRIVE_CLIENT_SECRET": "OneDrive Client Secret",
    "ONEDRIVE_TENANT_ID": "OneDrive Tenant ID",
    "ONEDRIVE_ACCESS_TOKEN": "OneDrive Access Token (optional)",
    "DB_CONN": "Database connection string (SQLAlchemy)",
    "MONGO_URI": "MongoDB URI",
    "API_TOKEN": "Generic API token (for REST API)",
}

def get_cred_value(key):
    sess_creds = st.session_state.get("creds", {})
    if key in sess_creds and sess_creds.get(key) not in (None, ""):
        return sess_creds.get(key)
    return os.getenv(key, "")


def save_creds_to_session(new_creds: dict):
    if "creds" not in st.session_state:
        st.session_state["creds"] = {}
    st.session_state["creds"].update(new_creds)
    st.success("Credentials saved to session (temporary).")


def save_creds_to_env(new_creds: dict, env_path: Optional[str] = None):
    env_file = find_dotenv()
    if not env_file:
        env_file = os.path.join(PROJECT_ROOT, ".env")
        open(env_file, "a").close()
    for k, v in new_creds.items():
        set_key(env_file, k, v)
    load_dotenv(override=True)
    st.success(f"Credentials written to {env_file}.")


# ----------------- Main App -----------------
def main():
    st.set_page_config(page_title='Smart NC Analyzer', layout='wide')
    st.title('Smart Non-Conformance Analyzer')

    try:
        init_db()
    except Exception as e:
        st.warning(f"Database init warning: {e}")

    # ---------------- Initialize session_state ----------------
    if "current_source" not in st.session_state:
        st.session_state.current_source = None
    if "source_changed" not in st.session_state:   # safety flag
        st.session_state.source_changed = False
    
    for key in ["raw_df", "df", "header_row", "logs", "current_log", "manual_saved",
                "processed", "embeddings", "labels", "creds"]:
        if key not in st.session_state:
            if key == "logs":
                st.session_state[key] = []
            elif key == "current_log":
                st.session_state[key] = 1
            elif key == "manual_saved":
                st.session_state[key] = False
            elif key == "creds":
                st.session_state[key] = {}
            else:
                st.session_state[key] = None
    
    # ---------------- Sidebar: Source + Auth settings ----------------
    source_choice = st.sidebar.selectbox(
        "Select input method",
        [
            "Upload File (CSV/Excel)",
            "Google Sheets",
            "OneDrive / SharePoint",
            "REST API (ERP/MES/QMS)",
            "SQL Database",
            "MongoDB",
            "Manual Entry",
        ],
        index=0,
        key="source_choice_widget",
    )
    
    # ---------------- Reset on source change ----------------
    if "last_source" not in st.session_state:
        st.session_state.last_source = source_choice
    
    if st.session_state.last_source != source_choice:
        # Clear all relevant session variables
        for key in ["raw_df", "df", "header_row", "logs", "current_log",
                    "manual_saved", "processed", "embeddings", "labels"]:
            if key == "logs":
                st.session_state[key] = []
            elif key == "current_log":
                st.session_state[key] = 1
            elif key == "manual_saved":
                st.session_state[key] = False
            else:
                st.session_state[key] = None
    
        # Clear sidebar widget states
        widget_keys_to_clear = [
            "uploaded_file", "sheet_url", "sa_input",
            "api_key_in", "use_service_account",
            "od_file", "od_token_ui", "od_client_id_ui", "od_client_secret_ui", "od_tenant_ui",
            "api_url", "api_token_ui", "extra_headers", "method",
            "db_conn_ui", "sql_query",
            "mongo_uri_ui", "mongo_db", "mongo_coll", "mongo_query_text",
        ]
        for wk in widget_keys_to_clear:
            if wk in st.session_state:
                del st.session_state[wk]
    
        # Update last_source
        st.session_state.last_source = source_choice
    
        # Rerun app to apply reset
        try:
            safe_rerun()
        except Exception:
            st.experimental_rerun()
    
    # ---------------- Sidebar: Credentials ----------------
    with st.sidebar.expander("🔒 Authentication & Credentials (expand to override)"):
        st.markdown("Credentials are loaded from environment variables by default. Use these fields to override for this session, or save to `.env` permanently.")
        cred_inputs = {}
        for k, label in CRED_KEYS.items():
            is_secret = "SECRET" in k or "TOKEN" in k or "PASSWORD" in k
            default = get_cred_value(k)
            cred_inputs[k] = st.text_input(label, value=default, key=f"cred_{k}", type="password" if is_secret else "default")
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save for Session Only"):
                session_pairs = {k: v for k, v in cred_inputs.items() if v}
                save_creds_to_session(session_pairs)
        with col2:
            if st.button("Save to .env Permanently"):
                env_pairs = {k: v for k, v in cred_inputs.items() if v}
                try:
                    save_creds_to_env(env_pairs)
                except Exception as e:
                    st.error(f"Failed to write to .env: {e}")
    
    # ----------------- Ingestion UI per source -----------------
    df = None
    st.sidebar.markdown("---")
    
    # Clear previous session data when switching source
    if "last_source" not in st.session_state:
        st.session_state.last_source = None
    if st.session_state.last_source != source_choice:
        for key in ["df", "raw_df", "uploaded_file_bytes"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_source = source_choice
    
    # ---- Upload File ----
    if source_choice == "Upload File (CSV/Excel)":
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded:
            try:
                df = ingest_file(uploaded)
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.raw_df = df
            except Exception as e:
                st.error(f"File ingestion failed: {e}")

    # Keep DataFrame across reruns
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df







    
    
    # ---- Google Sheets ----
    elif source_choice == "Google Sheets":
        st.sidebar.write("Google Sheets options")
        sheet_url = st.sidebar.text_input("Sheet URL or ID", value="", key="sheet_url")
    
        if st.sidebar.button("Load Google Sheet"):
            try:
                def extract_sheet_id(url_or_id: str) -> str:
                    if "/d/" in url_or_id:
                        return url_or_id.split("/d/")[1].split("/")[0]
                    return url_or_id
    
                sheet_id = extract_sheet_id(sheet_url)
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid=0"
                df = pd.read_csv(csv_url)
    
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.raw_df = df
                    st.session_state.source_changed = True
                    safe_rerun()
            except Exception as e:
                st.error(f"Google Sheets ingestion failed: {e}")
    
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
    
    
    # ---- OneDrive / SharePoint ----
    elif source_choice == "OneDrive / SharePoint":
        st.sidebar.write("OneDrive / SharePoint options")
        od_file = st.sidebar.text_input("File path or sharing URL", value="")
        od_token = get_cred_value("ONEDRIVE_ACCESS_TOKEN")
        od_client_id = get_cred_value("ONEDRIVE_CLIENT_ID")
        od_client_secret = get_cred_value("ONEDRIVE_CLIENT_SECRET")
        od_tenant = get_cred_value("ONEDRIVE_TENANT_ID")
    
        od_token_ui = st.sidebar.text_input("Access Token", value=od_token or "", type="password")
        od_client_id_ui = st.sidebar.text_input("Client ID", value=od_client_id or "")
        od_client_secret_ui = st.sidebar.text_input("Client Secret", value=od_client_secret or "", type="password")
        od_tenant_ui = st.sidebar.text_input("Tenant ID", value=od_tenant or "")
    
        if st.sidebar.button("Load from OneDrive"):
            try:
                df = ingest_onedrive(
                    od_file,
                    access_token=od_token_ui or od_token,
                    client_id=od_client_id_ui or od_client_id,
                    client_secret=od_client_secret_ui or od_client_secret,
                    tenant_id=od_tenant_ui or od_tenant,
                )
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.raw_df = df
                    st.session_state.source_changed = True
                    safe_rerun()
            except Exception as e:
                st.error(f"OneDrive ingestion failed: {e}")
    
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
    
    
    # ---- REST API ----
    elif source_choice == "REST API (ERP/MES/QMS)":
        st.sidebar.write("REST API options")
        api_url = st.sidebar.text_input("API Endpoint URL", value="")
        api_token = get_cred_value("API_TOKEN")
        api_token_ui = st.sidebar.text_input("API Token", value=api_token or "", type="password")
        extra_headers = st.sidebar.text_area("Additional headers (JSON)", value="{}")
        method = st.sidebar.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
    
        if st.sidebar.button("Fetch API"):
            try:
                headers = {}
                try:
                    headers = json.loads(extra_headers)
                except Exception:
                    st.warning("Invalid JSON headers; using {}")
                if api_token_ui:
                    headers.setdefault("Authorization", f"Bearer {api_token_ui}")
    
                df = ingest_rest_api(api_url, method=method, headers=headers)
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.raw_df = df
                    st.session_state.source_changed = True
                    safe_rerun()
            except Exception as e:
                st.error(f"REST API ingestion failed: {e}")
    
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
    
    
    # ---- SQL Database ----
    elif source_choice == "SQL Database":
        st.sidebar.write("SQL Database options")
        db_conn_env = get_cred_value("DB_CONN")
        db_conn_ui = st.sidebar.text_input("DB connection string", value=db_conn_env or "")
        sql_query = st.sidebar.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 100")
    
        if st.sidebar.button("Run Query"):
            conn_str = db_conn_ui or db_conn_env
            if not conn_str:
                st.error("No DB connection string supplied.")
            else:
                try:
                    df = ingest_database(conn_str, sql_query)
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.session_state.raw_df = df
                        st.session_state.source_changed = True
                        safe_rerun()
                except Exception as e:
                    st.error(f"Database ingestion failed: {e}")
    
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
    
    
    # ---- MongoDB ----
    elif source_choice == "MongoDB":
        st.sidebar.write("MongoDB options")
        mongo_uri_env = get_cred_value("MONGO_URI")
        mongo_uri_ui = st.sidebar.text_input("Mongo URI", value=mongo_uri_env or "")
        mongo_db = st.sidebar.text_input("Database name")
        mongo_coll = st.sidebar.text_input("Collection name")
        mongo_query_text = st.sidebar.text_area("Query (JSON)", value="{}")
    
        if st.sidebar.button("Load MongoDB"):
            try:
                q = {}
                try:
                    q = json.loads(mongo_query_text)
                except Exception:
                    st.warning("Invalid JSON query; using {}")
                df = ingest_mongodb(mongo_uri_ui or mongo_uri_env, mongo_db, mongo_coll, query=q)
    
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.raw_df = df
                    st.session_state.source_changed = True
                    safe_rerun()
            except Exception as e:
                st.error(f"MongoDB ingestion failed: {e}")
    
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
    
    
    # ---- Manual Entry ----
    elif source_choice == "Manual Entry":
        if not st.session_state.manual_saved:
            df = manual_log_entry()
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.raw_df = df
                st.session_state.manual_saved = True
                st.session_state.current_log = 1
                st.session_state.logs = []
                st.session_state.source_changed = True
                safe_rerun()
        else:
            df = st.session_state.df
    
        if df is not None:
            print(df.dtypes)

    
    # ---------------- Persist any loaded data ----------------
    if df is not None and source_choice != "Manual Entry":
        st.session_state.df = df.copy()
        st.session_state.raw_df = df.copy()


# ----------------- Data Preview and downstream workflow -----------------
 
    # Ensure DataFrame from manual logs is captured
    
    if df is not None:
        if isinstance(df, pd.DataFrame) and not df.empty:
        # Keep original uploaded dataframe intact
            if "raw_df_original" not in st.session_state or st.session_state.raw_df_original is None:
                st.session_state.raw_df_original = df.copy()

            # Store raw_df for reference
            st.session_state.raw_df = df

            # Initialize header_row once
        if "header_row" not in st.session_state or st.session_state.header_row is None:
                st.session_state.header_row = 0

         # Apply header using current selection (from pristine original)
        if not st.session_state.get("manual_df_ready"):
            st.session_state.df = apply_row_as_header(
                st.session_state.raw_df_original.copy(),
                st.session_state.header_row
            )
        else:
            st.session_state.df = df  # Manual logs use as-is

        st.success(
            f"Data loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns."
        )
    else:
        pass  # Removed warning



    # Main area: only show preview/analysis if raw_df present
       # Main area: only show preview/analysis if raw_df present
    if st.session_state.get("raw_df") is not None and not st.session_state.get("raw_df").empty:
        st.subheader("Data Preview & Actions")
    
        df = st.session_state.df
    
        # Header selector for uploaded/raw data (skip for manual entry)
     
        if st.session_state.get("input_type") != "Manual Entry":
            # Ensure we have a pristine copy of the originally uploaded data.
            # This will only be created once (the first time this block runs after upload).
            if "raw_df_original" not in st.session_state or st.session_state.raw_df_original is None:
                # store the original upload as the canonical source for header changes
                st.session_state.raw_df_original = st.session_state.raw_df.copy()
        
            max_row = len(st.session_state.raw_df_original) - 1
            new_header_row = st.number_input(
                "Row number to use as header (0-indexed)",
                min_value=0,
                max_value=max_row,
                value=int(st.session_state.header_row) if st.session_state.header_row is not None else 0,
                step=1,
                help="Pick a row from the file to become column headers."
            )
        
            if int(new_header_row) != int(st.session_state.header_row):
                st.session_state.header_row = int(new_header_row)
                # Always apply header on the pristine original uploaded dataframe
                st.session_state.df = apply_row_as_header(
                    st.session_state.raw_df_original.copy(),
                    st.session_state.header_row
                )
                df = st.session_state.df
                safe_rerun()




        

        # Tabs: Preview / Save
        tab1, tab2 = st.tabs(["Preview", "Save & Analyze"])

        with tab1:
            df_display = df.reset_index(drop=True).rename_axis("No").rename(lambda x: x + 1, axis=0)
            st.dataframe(df_display.head(100))

            # Quick save
            file_name = st.text_input("Save preview as filename", value="uploaded_data.parquet")
            if st.button("Save Preview"):
                try:
                    save_processed(df.copy(), file_name)
                    st.success(f"Preview saved to {file_name}")
                except Exception as e:
                    st.error(f"Failed to save preview: {e}")

        with tab2:
            # --- Preprocess & embed ---
            st.markdown("### Text Selection")
            object_cols = [c for c in df.columns if df[c].dtype == 'object']
            default_text_cols = object_cols[:2]
            text_cols = st.multiselect(
                'Text columns to use for embedding',
                options=df.columns.tolist(),
                default=default_text_cols
            )
            
            def preprocess_df_keepall(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
                """
                Keep ALL original columns as-is and add 'clean_text' for embeddings.
                """
                df_copy = df.copy()
            
                # make sure we only use valid columns
                valid_text_cols = [c for c in text_cols if c in df_copy.columns]
            
                if not valid_text_cols:
                    df_copy["clean_text"] = ""
                else:
                    df_copy["clean_text"] = (
                        df_copy[valid_text_cols]
                        .astype(str)
                        .apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)
                    )
            
                df_copy["clean_text"] = df_copy["clean_text"].str.strip().str.lower()
                return df_copy
            
            
            if st.button('Preprocess & Embed'):
                if not text_cols:
                    st.error("Please select at least one text column.")
                else:
                    try:
                        # --- Preprocess ---
                        p = preprocess_df_keepall(df, text_cols)
                        st.session_state['processed'] = p
                        st.success('Preprocessing complete')
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")
                        st.stop()
            
                    try:
                        # --- Compute embeddings ---
                        embeddings = embed_texts(p['clean_text'].tolist())
                        st.session_state['embeddings'] = embeddings
                        st.success('Embeddings computed')
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")
                        st.stop()
            
                    # --- Automatically run clustering ---
                    from PIL import Image as PILImage
                    st.subheader("Clustering & Visualization")
                    valid_p = isinstance(p, pd.DataFrame) and not p.empty
                    valid_embeddings = embeddings is not None and len(embeddings) > 0
            
                    if valid_p and valid_embeddings:
                        try:
                            from config import RANDOM_STATE
                            with st.spinner("Evaluating optimal clusters..."):
                                best, results = evaluate_kmeans(embeddings, k_values=list(range(2, 8)))
            
                            metrics_summary = {
                                "Silhouette Score": best["Silhouette Score"],
                                "Davies-Bouldin Score": best["Davies-Bouldin Score"],
                                "interpretation": best["interpretation"],
                            }
            
                            st.session_state['cluster_metrics'] = metrics_summary
                            st.session_state['cluster_labels'] = best["labels"]
                            st.session_state['cluster_fig'] = cluster_scatter(embeddings, best["labels"])
                            st.session_state['cluster_text'] = (
                                f"Best K={best['k']} | Silhouette={best['Silhouette Score']:.3f} | "
                                f"Davies-Bouldin={best['Davies-Bouldin Score']:.3f}"
                            )
            
                            # Save cluster chart as PNG
                            clusters_chart_path = "clusters_rgb.png"
                            st.session_state['cluster_fig'].write_image(
                                clusters_chart_path, format="png", scale=2, engine="kaleido"
                            )
                            img = PILImage.open(clusters_chart_path).convert("RGB")
                            img.save(clusters_chart_path)
                            st.session_state["clusters_chart"] = clusters_chart_path
            
                            # Save cluster summary text
                            clusters_summary = (
                                f"Best K={best['k']}, Silhouette={best['Silhouette Score']:.3f}, "
                                f"Davies-Bouldin={best['Davies-Bouldin Score']:.3f}. "
                                f"Interpretation: {best['interpretation']}"
                            )
                            st.session_state["clusters_summary"] = clusters_summary
            
                        except Exception as e:
                            st.error(f"Clustering failed: {e}")
            
            # --- Persistent display of clusters ---
            if "cluster_text" in st.session_state and \
               "cluster_metrics" in st.session_state and \
               "cluster_fig" in st.session_state:
                
                st.subheader("Clustering & Visualization")
                st.success(st.session_state['cluster_text'])
                st.info(st.session_state['cluster_metrics']["interpretation"])
                st.plotly_chart(st.session_state['cluster_fig'], use_container_width=True)

    
          
            # --- NLTK & Utilities ---

            # Make sure NLTK has the WordNet lemmatizer
            from nltk.stem import WordNetLemmatizer
            from collections import Counter
            import re
            from fuzzywuzzy import process  # pip install fuzzywuzzy[speedup]
            
            nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            
            def normalize_text(text):
                """
                Clean, lowercase, remove numbers, normalize spacing/hyphens, and lemmatize words (singular form).
                Safely handles None, NaN, non-string, or unexpected types.
                """
                if text is None:
                    return ""
                
                # Ensure it's a string; if not, attempt conversion
                if not isinstance(text, str):
                    try:
                        text = str(text)
                    except Exception:
                        return ""
            
                text = text.lower()
                text = re.sub(r"\d+", "", text)            # remove numbers
                text = re.sub(r"[^a-z\s-]", "", text)     # remove punctuation except hyphen
                text = re.sub(r"[-_]", " ", text)         # replace hyphen/underscore with space
                text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
                tokens = text.split()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
                return " ".join(tokens).strip()

            def find_recurring_issues(df, top_n=10, similarity_threshold=80):
                """
                Detect recurring issues in columns related to issues, problems, defects, faults.
                Normalize, merge similar phrases, pick the most descriptive phrase (longest), 
                capitalize first letter, and return top N.
                """
                issue_synonyms = ["issue", "issues", "problem", "problems", "defect", "defects", "fault", "faults"]
            
                # find candidate columns
                issue_cols = [col for col in df.columns if any(syn in col.lower() for syn in issue_synonyms)]
                if not issue_cols:
                    return {}
            
                all_issues = []
                for col in issue_cols:
                    all_issues.extend(df[col].dropna().tolist())  # keep original type for length comparison
            
                # normalize
                normalized = [normalize_text(t) for t in all_issues if t and str(t).strip()]
            
                # merge similar issues
                merged_issues_dict = {}  # key: representative issue, value: count
                for orig, norm in zip(all_issues, normalized):
                    if not merged_issues_dict:
                        merged_issues_dict[norm] = 1
                    else:
                        # find best match among existing merged issues
                        match, score = process.extractOne(norm, list(merged_issues_dict.keys()))
                        if score >= similarity_threshold:
                            existing_count = merged_issues_dict[match]  # get existing count
                            # pick the longer/original issue as representative
                            candidate_phrase = orig if len(orig.split()) > len(match.split()) else match
                            # remove old key if different
                            if candidate_phrase != match:
                                merged_issues_dict.pop(match)
                            # update count correctly
                            merged_issues_dict[candidate_phrase] = existing_count + 1
                        else:
                            merged_issues_dict[norm] = 1
            
                # count frequency and pick top N
                counter = Counter(merged_issues_dict)
                top_issues = dict(counter.most_common(top_n))
            
                # Capitalize first letter
                top_issues_cap = {k.capitalize(): v for k, v in top_issues.items()}
            
                return top_issues_cap


         
            # ---------------------------
            # Pareto chart from recurring issues table
           # ---------------------------
            # --- Recurring Issues & Pareto ---
            from PIL import Image as PILImage

            st.subheader("Recurring Issues & Pareto Analysis")
            
            p = st.session_state.get("processed")
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                # --- Find recurring issues ---
                recurring_issues = find_recurring_issues(p, top_n=10)
            
                if recurring_issues:
                    # Convert to DataFrame for display and plotting
                    data = [{"Issue": k, "Occurrences": v} for k, v in recurring_issues.items()]
                    recurring_df = pd.DataFrame(data)
                    recurring_df.index = recurring_df.index + 1
                    recurring_df.index.name = "S/N"
            
                    st.markdown(" ")
                    st.table(recurring_df)
            
                    # --- Pareto Table from Recurring Issues ---
                    pareto_df = recurring_df.copy()
                    pareto_df["Percent"] = (pareto_df["Occurrences"] / pareto_df["Occurrences"].sum() * 100).round(2)
                    pareto_df["Cumulative %"] = pareto_df["Percent"].cumsum().round(2)
            
                    # --- Save tables for PDF ---
                    st.session_state["recurring_issues_df"] = recurring_df
                    st.session_state["pareto_df"] = pareto_df
                    st.session_state["pareto_summary"] = (
                        f"Top recurring issues Pareto analysis completed. Top issue: {pareto_df.iloc[0]['Issue']}."
                    )
            
                    # --- Plot Pareto using Plotly ---
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_bar(
                        x=pareto_df['Issue'],
                        y=pareto_df['Occurrences'],
                        name='Occurrences',
                        marker_color='teal'
                    )
                    fig.add_scatter(
                        x=pareto_df['Issue'],
                        y=pareto_df['Cumulative %'],
                        name='Cumulative %',
                        yaxis='y2',
                        marker_color='crimson'
                    )
            
                    fig.update_layout(
                        title="Pareto Chart of Top Recurring Issues",
                        width=1400,   # wider chart
                        height=800,   # taller chart
                        margin=dict(l=80, r=80, t=100, b=250),  # extra bottom margin for long labels
                        yaxis=dict(title='Occurrences'),
                        yaxis2=dict(title='Cumulative %', overlaying='y', side='right'),
                        xaxis=dict(
                            tickangle=-45,
                            tickfont=dict(size=12),
                            automargin=True
                        ),
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
            
                    st.plotly_chart(fig, use_container_width=True)
            
                    # --- Save chart for PDF ---
                    pareto_chart_path = "pareto_rgb.png"
                    fig.write_image(pareto_chart_path, format="png", scale=2, engine="kaleido")
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(pareto_chart_path).convert("RGB")
                    pil_img.save(pareto_chart_path)
                    st.session_state["pareto_chart"] = pareto_chart_path
            
                else:
                    st.info("No recurring issues detected to plot Pareto.")
            else:
                st.warning("No processed data available for recurring issues/Pareto analysis.")



            # --- SPC Section ---
            # --- SPC Section ---
            st.subheader("Statistical Process Control (SPC)")
            p = st.session_state.get('processed')
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    spc_df = p.copy()
                    # Convert all numeric-like columns
                    for c in spc_df.columns:
                        if not pd.api.types.is_numeric_dtype(spc_df[c]):
                            spc_df[c] = pd.to_numeric(spc_df[c], errors='coerce')
            
                    num_cols = [c for c in spc_df.select_dtypes(include=['number']).columns if spc_df[c].notna().any()]
                    if not num_cols:
                        st.info("No numeric columns available for SPC analysis after conversion.")
                    else:
                        spc_col_selected = st.selectbox('Select numeric column for SPC', options=num_cols, key='spc_col_select')
                        subgroup_size = st.number_input('Subgroup Size (1 = I-MR chart)', min_value=1, value=1, key='spc_subgroup')
            
                        # Optional time columns
                        time_cols = [c for c in spc_df.columns if pd.api.types.is_datetime64_any_dtype(spc_df[c])]
                        for c in spc_df.select_dtypes(include=['object']).columns:
                            try:
                                spc_df[c] = pd.to_datetime(spc_df[c], errors='coerce')
                                if spc_df[c].notna().any():
                                    time_cols.append(c)
                            except Exception:
                                continue
                        time_col_selected = st.selectbox('Optional time column', options=[None] + time_cols, key='spc_time_col_select')
            
                        # --- Run + Reset Buttons ---
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            run_clicked = st.button("Run SPC Analysis")
                        with col2:
                            reset_clicked = st.button("Reset SPC Analysis", type="secondary")
            
                        # Run logic
                        if run_clicked:
                            try:
                                from visualization import plot_spc_chart
                                fig_spc = plot_spc_chart(spc_df, spc_col_selected, subgroup_size=subgroup_size, time_col=time_col_selected)
                                st.session_state['spc_fig'] = fig_spc
                                st.session_state['spc_col_saved'] = spc_col_selected
            
                                # Save chart for PDF
                                spc_chart_path = "spc_chart.png"
                                fig_spc.write_image(spc_chart_path, format="png", scale=2, engine="kaleido")
                                img = PILImage.open(spc_chart_path).convert("RGB")
                                img.save(spc_chart_path)
                                st.session_state["spc_chart"] = spc_chart_path
            
                                # Summary
                                spc_summary = "Process shows 2 points outside control limits; needs investigation."
                                st.session_state["spc_summary"] = spc_summary
            
                            except Exception as e:
                                st.error(f"SPC plotting failed: {e}")
            
                        # Reset logic
                        if reset_clicked:
                            for key in ["spc_fig", "spc_col_saved", "spc_chart", "spc_summary"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success("SPC analysis has been reset. Please run again.")
            
                except Exception as e:
                    st.error(f"SPC setup failed: {e}")
            
            else:
                st.warning("No processed data available for SPC. Please preprocess first.")
            
            # --- Persistent SPC Display ---
            if "spc_fig" in st.session_state and "spc_col_saved" in st.session_state:
                st.success(f"SPC Chart for: {st.session_state.get('spc_col_saved', '')}")
                st.plotly_chart(
                    st.session_state['spc_fig'],
                    use_container_width=True,
                    key=f"spc_chart_{st.session_state.get('spc_col_saved', '')}"
                )



        
            # --- Global Date Format Selector ---
            st.subheader("🗓️ Date Format Settings")
            
            format_options = {
                "Auto-detect": None,
                "YYYY-MM-DD (2025-09-04)": "%Y-%m-%d",
                "YYYY-DD-MM (2025-04-09)": "%Y-%d-%m",
                "DD-MM-YYYY (04-09-2025)": "%d-%m-%Y",
                "MM-DD-YYYY (09-04-2025)": "%m-%d-%Y",
                "DD/MM/YYYY (04/09/2025)": "%d/%m/%Y",
                "MM/DD/YYYY (09/04/2025)": "%m/%d/%Y",
                "YYYY/MM/DD (2025/09/04)": "%Y/%m/%d",
            }
            fmt_choice = st.selectbox("Select date format for all dashboards", options=list(format_options.keys()))
            st.session_state["date_format"] = format_options[fmt_choice]
            
            
            def parse_dates_strict(series, date_format=None, sample_size=5):
                """
                Strictly parse a pandas Series of dates, with support for:
                - Auto-detect mode (scan first few rows)
                - Mixed formats normalization into a single chosen format
                
                Args:
                    series: pandas Series of strings
                    date_format: optional strftime format (from UI)
                    sample_size: rows to test for auto-detect
                
                Returns:
                    parsed_series: pandas Series of datetime64
                    diagnostics: dict with parse stats
                """
                raw = series.astype(str).str.strip()
            
                # --- 1. Auto-detect if no format selected ---
                fmt_used = None
                if not date_format:
                    sample = raw.head(sample_size).dropna()
                    parsed_test1 = pd.to_datetime(sample, format="%Y-%m-%d", errors="coerce")
                    parsed_test2 = pd.to_datetime(sample, format="%Y-%d-%m", errors="coerce")
            
                    success1 = parsed_test1.notna().sum()
                    success2 = parsed_test2.notna().sum()
            
                    if success1 >= success2:
                        fmt_used = "%Y-%m-%d"
                    else:
                        fmt_used = "%Y-%d-%m"
                else:
                    fmt_used = date_format
            
                # --- 2. Try primary format first ---
                parsed = pd.to_datetime(raw, format=fmt_used, errors="coerce")
            
                # --- 3. Handle mixed formats (fallback attempts) ---
                if parsed.isna().any():
                    # Try alternate common formats only on failed rows
                    alt_formats = [
                        "%Y-%m-%d",
                        "%Y-%d-%m",
                        "%d-%m-%Y",
                        "%m-%d-%Y",
                        "%d/%m/%Y",
                        "%m/%d/%Y",
                        "%Y/%m/%d",
                    ]
                    failed_idx = parsed[parsed.isna()].index
            
                    for fmt in alt_formats:
                        parsed_alt = pd.to_datetime(raw.loc[failed_idx], format=fmt, errors="coerce")
                        parsed.loc[failed_idx] = parsed.loc[failed_idx].fillna(parsed_alt)
            
                # --- 4. Normalize into final selected format ---
                # NOTE: The datetime64 values are stored consistently,
                # but for display/export you can enforce formatting later:
                # parsed.dt.strftime(date_format)
                
                diagnostics = {
                    "total_rows": len(raw),
                    "parsed_count": int(parsed.notna().sum()),
                    "failed_count": int(parsed.isna().sum()),
                    "failed_examples": raw[parsed.isna()].unique()[:5].tolist(),
                    "format_used": fmt_used,
                    "note": "Mixed formats auto-converted into chosen format"
                }
            
                return parsed, diagnostics



            # --- Trend Dashboard ---
            # --- Trend Dashboard ---
            st.subheader("📈 Trend Dashboard")
            p = st.session_state.get("processed")
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    trend_df = p.copy()
            
                    # Convert numeric columns safely
                    for c in trend_df.columns:
                        if trend_df[c].dtype == "object":
                            trend_df[c] = pd.to_numeric(
                                trend_df[c].astype(str).str.replace(",", "").str.strip(),
                                errors="ignore"
                            )
            
                    # Detect numeric columns
                    num_cols = [
                        c for c in trend_df.select_dtypes(include=["number"]).columns
                        if trend_df[c].notna().any()
                    ]
            
                    # Detect date columns using global date format
                    date_cols = []
                    for c in trend_df.columns:
                        try:
                            parsed, diag = parse_dates_strict(trend_df[c], st.session_state.get("date_format"))
                            if parsed.notna().any():
                                trend_df[c] = parsed  # overwrite with parsed dates
                                date_cols.append(c)
                        except Exception:
                            continue
            
                    # --- Trend Analysis ---
                    if date_cols and num_cols:
                        date_col = st.selectbox("Select Date Column", options=date_cols, key="trend_date_col")
                        value_col = st.selectbox("Select Value Column", options=num_cols, key="trend_value_col")
            
                        # --- Run + Reset Buttons ---
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            run_trend = st.button("Run Trend Analysis", key="trend_btn")
                        with col2:
                            reset_trend = st.button("Reset Trend Analysis", type="secondary", key="trend_reset_btn")
            
                        if run_trend:
                            try:
                                trend_df[date_col], diag = parse_dates_strict(
                                    trend_df[date_col],
                                    st.session_state.get("date_format")
                                )
            
                                fig_trend = plot_trend_dashboard(
                                    trend_df,
                                    date_col=date_col,
                                    value_col=value_col,
                                    date_format=st.session_state.get("date_format")
                                )
            
                                if fig_trend:
                                    st.session_state["trend_fig"] = fig_trend
                                    st.session_state["trend_col"] = value_col
                                    st.session_state["trend_date_col_saved"] = date_col
            
                                    trend_chart_path = "trend_chart.png"
                                    try:
                                        fig_trend.write_image(trend_chart_path, format="png", scale=2, engine="kaleido")
                                        PILImage.open(trend_chart_path).convert("RGB").save(trend_chart_path)
                                        st.session_state["trend_chart"] = trend_chart_path
                                        st.session_state["trend_summary"] = (
                                            f"Trend chart of '{value_col}' over '{date_col}'"
                                        )
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not save trend chart image: {e}")
                                        st.session_state["trend_chart"] = None
            
                            except Exception as e:
                                st.warning(f"⚠️ Unable to render trend plot: {e}")
            
                        # Reset logic
                        if reset_trend:
                            for key in ["trend_fig", "trend_col", "trend_date_col_saved", "trend_chart", "trend_summary"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success("Trend analysis has been reset. Please run again.")
            
                    else:
                        st.info("No valid date and numeric column pair available for trend plotting.")
            
                    # --- Persistent Display ---
                    if "trend_fig" in st.session_state and "trend_col" in st.session_state:
                        st.success(
                            f"Trend Chart: {st.session_state.get('trend_col')} "
                            f"over {st.session_state.get('trend_date_col_saved')}"
                        )
                        st.plotly_chart(st.session_state["trend_fig"], use_container_width=True)
            
                    # --- Time-Series Analysis ---
                    st.subheader("⏳ Time-Series Trend Analysis")
                    if date_cols and num_cols:
                        time_col = st.selectbox("Select time column", options=date_cols, key="time_col")
                        value_col = st.selectbox("Select value column", options=num_cols, key="time_value_col")
            
                        freq_options = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
                        freq_choice = st.selectbox("Select aggregation level", options=list(freq_options.keys()))
                        agg_options = ["mean", "sum", "max", "min"]
                        agg_choice = st.selectbox("Select aggregation function", options=agg_options)
            
                        # --- Run + Reset Buttons ---
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            run_time = st.button("Plot Time-Series Trend", key="time_btn")
                        with col2:
                            reset_time = st.button("Reset Time-Series Trend", type="secondary", key="time_reset_btn")
            
                        if run_time:
                            try:
                                parsed, diag = parse_dates_strict(
                                    p[time_col],
                                    st.session_state.get("date_format")
                                )
            
                                if diag["parsed_count"] > 0:
                                    ts_df = p.copy()
                                    ts_df["_parsed_time"] = parsed
            
                                    fig_time = plot_time_series_trend(
                                        ts_df,
                                        date_col="_parsed_time",
                                        value_col=value_col,
                                        freq=freq_options[freq_choice],
                                        agg_func=agg_choice,
                                        date_format=st.session_state.get("date_format")
                                    )
            
                                    if fig_time:
                                        st.session_state["time_fig"] = fig_time
                                        st.session_state["time_col"] = value_col
                                        st.session_state["time_date_col_saved"] = time_col
            
                                        time_chart_path = "time_series_trend.png"
                                        try:
                                            fig_time.write_image(time_chart_path, format="png", scale=2, engine="kaleido")
                                            PILImage.open(time_chart_path).convert("RGB").save(time_chart_path)
                                            st.session_state["time_chart"] = time_chart_path
                                            st.session_state["time_summary"] = (
                                                f"{freq_choice} trend of '{value_col}' over '{time_col}', aggregated by {agg_choice}"
                                            )
                                        except Exception as e:
                                            st.warning(f"⚠️ Could not save time-series chart image: {e}")
                                            st.session_state["time_chart"] = None
                            except Exception as e:
                                st.warning(f"⚠️ Error generating time-series: {e}")
            
                        # --- Reset Logic ---
                        if reset_time:
                            for key in ["time_fig", "time_col", "time_date_col_saved", "time_chart", "time_summary"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success("Time-Series trend has been reset. Please run again.")
            
                    else:
                        st.warning("No valid datetime and numeric column pair for time-series analysis.")
            
                    # --- Persistent display for Time-Series ---
                    if "time_fig" in st.session_state and "time_col" in st.session_state:
                        st.success(
                            f"Time-Series Chart: {st.session_state.get('time_col')} "
                            f"over {st.session_state.get('time_date_col_saved')}"
                        )
                        st.plotly_chart(st.session_state['time_fig'], use_container_width=True)
            
                except Exception as e:
                    st.error(f"Trend setup failed: {e}")




            # --- Root Cause Analysis (RCA) --

            # --- Page Setup ---
            st.set_page_config(page_title="AI-Powered RCA", layout="wide")
            st.title("🛠️ AI-Powered Root Cause Analysis")
            st.markdown(
                "This RCA tool uses your preprocessed data and a reference folder of past issues "
                "to generate RCA, 5-Whys, CAPA, and Fishbone diagrams."
            )
            st.markdown("---")
            
            
            
            
            # --- Processed table session ---
            if p is not None and not p.empty:
                idx = st.number_input(
                    "Pick row index to analyze",
                    min_value=0,
                    max_value=len(p) - 1,
                    value=0,
                )
                row = p.iloc[int(idx)]
                raw_text = str(row.get("combined_text") or row.get("clean_text") or "")
                st.markdown("**Selected row preview:**")
                st.write(raw_text)
            else:
                st.warning("No processed data available to analyze.")
            


    
            # --- RCA mode (fixed to AI) ---
            mode = "AI-Powered (LLM+Agent)"
            st.markdown("**RCA Mode:** AI-Powered (LLM+Agent)")
            
            # --- Run RCA ---
            if st.button("Run RCA"):
                with st.spinner("Running RCA using reference folder and AI agent..."):
                    try:
                        # Base directory of this script
                        current_dir = os.path.dirname(os.path.abspath(__file__))
            
                        # Possible data folder locations
                        possible_folders = [
                            os.path.join(current_dir, "..", "data", "processed"),
                            os.path.join(current_dir, "..", "main", "data", "processed"),
                            os.path.join(os.getcwd(), "NCA", "data", "processed"),
                            os.path.join(os.getcwd(), "nca", "data", "processed"),
                            os.path.join(os.getcwd(), "NCA", "main", "data", "processed"),
                            os.path.join(os.getcwd(), "nca", "main", "data", "processed"),
                        ]
                        reference_folder = next((f for f in possible_folders if os.path.exists(f)), None)
            
                        if not reference_folder:
                            st.warning("⚠️ Reference folder not found. Please create `NCA/data/` and add past RCA files.")
                            st.session_state["rca_result"] = {"error": "Reference folder missing."}
                        else:
                            st.success(f"📂 Using reference folder: {reference_folder}")
            
                            llm_backend = "gemini"

                            # Call RCA engine with dynamic backend
                            result = ai_rca_with_fallback(
                                record={"issue": raw_text},
                                processed_df=p,
                                sop_library=None,
                                qc_logs=None,
                                reference_folder=reference_folder,
                                llm_backend=llm_backend
                            )
                            st.session_state["rca_result"] = result
            
                    except Exception as e:
                        st.session_state["rca_result"] = {"error": str(e)}

            
            # --- RCA Results ---
            # --- RCA Results ---
            from reportlab.platypus import Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            styles = getSampleStyleSheet()
            
            result = st.session_state.get("rca_result", {})
            if result:
            
                # Initialize PDF content lists
                rca_pdf_content = []      # for PDF (Paragraph objects)
                rca_pdf_text_content = [] # for HTML/text joining
            
                # --- RCA - Details ---
                st.markdown("### RCA - Details")
            
                # --- Error Handling ---
                if result.get("error"):
                    st.error(result.get("error"))
                    rca_pdf_content.append(Paragraph(f"Error: {result.get('error')}", styles['Normal']))
                    rca_pdf_text_content.append(f"Error: {result.get('error')}")
            
                # --- 5-Whys Analysis ---
                why = result.get("why_analysis") or result.get("five_whys")
                if why:
                    st.markdown("**5-Whys Analysis:**")
                    rca_pdf_content.append(Paragraph("5-Whys Analysis:", styles['Heading3']))
                    rca_pdf_text_content.append("5-Whys Analysis:")
            
                    if isinstance(why, list):
                        for i, w in enumerate(why, start=1):
                            st.write(f"{i}. {w}")
                            rca_pdf_content.append(Paragraph(f"{i}. {w}", styles['Normal']))
                            rca_pdf_text_content.append(f"{i}. {w}")
                    else:
                        st.write(why)
                        rca_pdf_content.append(Paragraph(str(why), styles['Normal']))
                        rca_pdf_text_content.append(str(why))
            
                # --- Root Cause ---
                if result.get("root_cause"):
                    st.markdown("**Root Cause:**")
                    st.write(result["root_cause"])
                    rca_pdf_content.append(Paragraph("Root Cause:", styles['Heading3']))
                    rca_pdf_content.append(Paragraph(result["root_cause"], styles['Normal']))
                    rca_pdf_text_content.append(f"Root Cause: {result['root_cause']}")
            
                # --- CAPA Recommendations ---
                capa = result.get("capa")
                if capa:
                    st.markdown("**CAPA Recommendations:**")
                    rca_pdf_content.append(Paragraph("Corrective and Preventive Actions (CAPA):", styles['Heading3']))
                    rca_pdf_text_content.append("Corrective and Preventive Actions (CAPA):")
            
                    if isinstance(capa, list):
                        for c in capa:
                            line = (
                                f"- {c.get('type', '')}: {c.get('action', '')} "
                                f"(Owner: {c.get('owner', 'Unassigned')}, Due: {c.get('due_in_days', '?')} days)"
                            )
                            st.write(line)
                            rca_pdf_content.append(Paragraph(line, styles['Normal']))
                            rca_pdf_text_content.append(line)
                    else:
                        st.write(capa)
                        rca_pdf_content.append(Paragraph(str(capa), styles['Normal']))
                        rca_pdf_text_content.append(str(capa))
            
                # --- Fallback: Raw AI Report ---
                raw_text = None
                if not any([why, result.get("root_cause"), capa]):
                    raw_text = result.get("parsed", {}).get("raw_text") or result.get("response")
                    if raw_text:
                        st.markdown("**AI RCA Report:**")
                        st.markdown(raw_text)
            
                        # Convert AI markdown into structured flowables (headings, lists, tables)
                        converted_content = convert_markdown_to_pdf_content(raw_text, styles)
                        st.session_state["rca_pdf_content"] = converted_content
                        st.session_state["rca_pdf_text"] = raw_text
            
                        # Categorize for fishbone (if AI didn’t provide structured fishbone)
                        st.session_state["fishbone_categories"] = result.get("fishbone") or {}
            
                # --- Fishbone Visualization Section ---
                points = extract_main_points(result, raw_text)
                fishbone_data = categorize_6m(points)
                
                # Save for UI + PDF
                st.session_state["fishbone_data"] = fishbone_data
                
                # Show in UI
                st.markdown("### Fishbone Diagram")
                fig = visualize_fishbone_plotly(fishbone_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save as image for PDF
                fig_path = "/tmp/fishbone.png"
                fig.write_image(fig_path)
                st.session_state["fishbone_img"] = fig_path






            

            # Helper to fix black & white issue
            from PIL import Image as PILImage


            def rgb_image_for_pdf(path, width=400, height=250):
                """Convert any chart image into true RGB and return ReportLab-safe Image."""
                pil_img = PILImage.open(path).convert("RGB")   # force RGB, drop alpha
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")         # re-save as RGB PNG
                img_buffer.seek(0)
                return RLImage(img_buffer, width=width, height=height)

            # Main PDF generator
            def generate_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = []
            
                # Title
                elements.append(Paragraph("Smart Non-Conformance Analyzer Report", styles['Title']))
                elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                elements.append(Spacer(1, 20))
            
                # =====================
                # Clustering
                # =====================
                if "clusters_summary" in st.session_state:
                    elements.append(Paragraph("Clustering & Visualization", styles['Heading2']))
                    elements.append(Paragraph(st.session_state["clusters_summary"], styles['Normal']))
                    if "clusters_chart" in st.session_state:
                        elements.append(rgb_image_for_pdf(st.session_state["clusters_chart"]))

                    elements.append(Spacer(1, 20))


                # =====================
                # Recurring Issues Table
                # =====================
                if "recurring_issues_df" in st.session_state:
                    elements.append(Paragraph("Top Recurring Issues", styles['Heading2']))
                    recurring_df = st.session_state["recurring_issues_df"]
                
                    # Convert DataFrame to ReportLab Table
                    from reportlab.platypus import Table, TableStyle
                    table_data = [recurring_df.columns.tolist()] + recurring_df.reset_index().values.tolist()
                    tbl = Table(table_data, hAlign='LEFT')
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                        ('ALIGN',(0,0),(-1,-1),'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ]))
                    elements.append(tbl)
                    elements.append(Spacer(1, 20))

                
                # =====================
                # Pareto
                # =====================
                if "pareto_summary" in st.session_state:
                    elements.append(Paragraph("Pareto Analysis", styles['Heading2']))
                    elements.append(Paragraph(st.session_state["pareto_summary"], styles['Normal']))
                    if "pareto_chart" in st.session_state:
                        elements.append(rgb_image_for_pdf(st.session_state["pareto_chart"]))
                    elements.append(Spacer(1, 20))
            
                # =====================
                # SPC
                # =====================
                if "spc_summary" in st.session_state:
                    elements.append(Paragraph("Statistical Process Control (SPC)", styles['Heading2']))
                    elements.append(Paragraph(st.session_state["spc_summary"], styles['Normal']))
                    if "spc_chart" in st.session_state:
                        elements.append(rgb_image_for_pdf(st.session_state["spc_chart"]))
                    elements.append(Spacer(1, 20))
            
                # =====================
                # Trendline & Time-Series
                # =====================
                if "trend_chart" in st.session_state:
                    elements.append(Paragraph("Trendline Chart", styles['Heading2']))
                    elements.append(Paragraph(st.session_state.get("trend_summary", ""), styles['Normal']))
                    elements.append(rgb_image_for_pdf(st.session_state["trend_chart"]))
                    elements.append(Spacer(1, 20))
                
                # --- Time-Series Export ---
                if "time_chart" in st.session_state and st.session_state["time_chart"]:
                    elements.append(Paragraph("Time-Series Chart", styles['Heading2']))
                    elements.append(Paragraph(st.session_state.get("time_summary", ""), styles['Normal']))
                    elements.append(rgb_image_for_pdf(st.session_state["time_chart"]))
                    elements.append(Spacer(1, 20))
                
                elif "time_fig" in st.session_state:
                    # fallback if PNG not available
                    tmp_time_path = "tmp_time_series.png"
                    st.session_state["time_fig"].write_image(tmp_time_path, format="png", scale=2, engine="kaleido")
                    elements.append(Paragraph("Time-Series Chart", styles['Heading2']))
                    elements.append(Paragraph(st.session_state.get("time_summary", ""), styles['Normal']))
                    elements.append(rgb_image_for_pdf(tmp_time_path))
                    elements.append(Spacer(1, 20))

               
                # =====================
                # Root Cause Analysis (RCA)
                # =====================
                if "rca_pdf_content" in st.session_state and st.session_state["rca_pdf_content"]:
                    elements.append(Paragraph("Root Cause Analysis (RCA)", styles['Heading2']))
                    for para in st.session_state["rca_pdf_content"]:
                        elements.append(para)
                    elements.append(Spacer(1, 20))


                # =====================
                # Fishbone Diagram
                # =====================
                if "fishbone_img" in st.session_state:
                    elements.append(Paragraph("Fishbone Diagram", styles['Heading2']))
                    elements.append(Image(st.session_state["fishbone_img"], width=500, height=300))
                    elements.append(Spacer(1, 20))





            
                # Build PDF
                doc.build(elements)
                buffer.seek(0)
                return buffer




            if st.button("📄 Generate Report"):
                pdf_buffer = generate_pdf()
                st.download_button(
                    label="📥 Download Full Report (PDF)",
                    data=pdf_buffer,
                    file_name="SNCA_Report.pdf",
                    mime="application/pdf"
            )





if __name__ == "__main__":
    main()
