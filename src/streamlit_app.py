# ================================
# File: src/streamlit_app.py
# ================================
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np 
from typing import Optional, Dict
from dotenv import load_dotenv, set_key, find_dotenv

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
from rca_engine import rule_based_rca_suggestions, ai_rca_with_fallback
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
    )

    # ---------------- Reset on source change ----------------
    if st.session_state.current_source != source_choice:
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
        # Update current source
        st.session_state.current_source = source_choice
        # Rerun app to start clean
        safe_rerun()

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

    if source_choice == "Upload File (CSV/Excel)":
        uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded:
            try:
                df = ingest_file(uploaded)
            except Exception as e:
                st.error(f"File ingestion failed: {e}")

    elif source_choice == "Google Sheets":
        st.sidebar.write("Google Sheets options")
        sheet_url = st.sidebar.text_input("Sheet URL or ID", value="")
        sa_path = get_cred_value("GOOGLE_SERVICE_ACCOUNT_JSON")
        api_key = get_cred_value("GOOGLE_API_KEY")
        use_service_account = st.sidebar.checkbox("Use service account JSON (preferred)", value=bool(sa_path))
        if use_service_account:
            sa_input = st.sidebar.text_input("Service account JSON path (or leave to use env var)", value=sa_path or "")
            if st.sidebar.button("Load Google Sheet"):
                try:
                    df = ingest_google_sheet(sa_input or sa_path or sheet_url, service_account_json_path=sa_input or sa_path, api_key=api_key)
                except Exception as e:
                    st.error(f"Google Sheets ingestion failed: {e}")
        else:
            # CSV export mode
            api_key_in = st.sidebar.text_input("Optional: Google API Key (for public sheets)", value=api_key or "")
            if st.sidebar.button("Load Google Sheet (CSV export)"):
                try:
                    df = ingest_google_sheet(sheet_url, service_account_json_path=None, api_key=api_key_in)
                except Exception as e:
                    st.error(f"Google Sheets CSV ingestion failed: {e}")

    elif source_choice == "OneDrive / SharePoint":
        st.sidebar.write("OneDrive / SharePoint options")
        od_file = st.sidebar.text_input("File path or sharing URL", value="")
        od_token = get_cred_value("ONEDRIVE_ACCESS_TOKEN")
        od_client_id = get_cred_value("ONEDRIVE_CLIENT_ID")
        od_client_secret = get_cred_value("ONEDRIVE_CLIENT_SECRET")
        od_tenant = get_cred_value("ONEDRIVE_TENANT_ID")

        # allow overriding in UI
        od_token_ui = st.sidebar.text_input("Access Token (optional, short-lived)", value=od_token or "", type="password")
        od_client_id_ui = st.sidebar.text_input("Client ID (if using client credentials)", value=od_client_id or "")
        od_client_secret_ui = st.sidebar.text_input("Client Secret (if using client credentials)", value=od_client_secret or "", type="password")
        od_tenant_ui = st.sidebar.text_input("Tenant ID (if using client credentials)", value=od_tenant or "")

        if st.sidebar.button("Load from OneDrive"):
            try:
                df = ingest_onedrive(
                    od_file,
                    access_token=od_token_ui or od_token or None,
                    client_id=od_client_id_ui or od_client_id or None,
                    client_secret=od_client_secret_ui or od_client_secret or None,
                    tenant_id=od_tenant_ui or od_tenant or None,
                )
            except Exception as e:
                st.error(f"OneDrive ingestion failed: {e}")

    elif source_choice == "REST API (ERP/MES/QMS)":
        st.sidebar.write("REST API options")
        api_url = st.sidebar.text_input("API Endpoint URL", value="")
        api_token = get_cred_value("API_TOKEN")
        api_token_ui = st.sidebar.text_input("API Token (optional)", value=api_token or "", type="password")
        extra_headers = st.sidebar.text_area("Additional headers (JSON)", value="{}")
        method = st.sidebar.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
        if st.sidebar.button("Fetch API"):
            try:
                headers = {}
                try:
                    headers = json.loads(extra_headers)
                except Exception:
                    st.warning("Invalid JSON for extra headers; ignoring.")
                    headers = {}
                if api_token_ui:
                    headers.setdefault("Authorization", f"Bearer {api_token_ui}")
                df = ingest_rest_api(api_url, method=method, headers=headers)
            except Exception as e:
                st.error(f"REST API ingestion failed: {e}")

    elif source_choice == "SQL Database":
        st.sidebar.write("SQL Database options")
        db_conn_env = get_cred_value("DB_CONN")
        db_conn_ui = st.sidebar.text_input("DB connection string (or leave to use env DB_CONN)", value=db_conn_env or "")
        sql_query = st.sidebar.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 100")
        if st.sidebar.button("Run Query"):
            conn_str = db_conn_ui or db_conn_env
            if not conn_str:
                st.error("No DB connection string supplied (env DB_CONN or enter here).")
            else:
                try:
                    df = ingest_database(conn_str, sql_query)
                except Exception as e:
                    st.error(f"Database ingestion failed: {e}")

    elif source_choice == "MongoDB":
        st.sidebar.write("MongoDB options")
        mongo_uri_env = get_cred_value("MONGO_URI")
        mongo_uri_ui = st.sidebar.text_input("Mongo URI (or leave to use env MONGO_URI)", value=mongo_uri_env or "")
        mongo_db = st.sidebar.text_input("Database name")
        mongo_coll = st.sidebar.text_input("Collection name")
        mongo_query_text = st.sidebar.text_area("Query (JSON)", value="{}")
        if st.sidebar.button("Load MongoDB"):
            try:
                q = {}
                try:
                    q = json.loads(mongo_query_text)
                except Exception:
                    st.warning("Invalid JSON for Mongo query; using empty query {}.")
                    q = {}
                df = ingest_mongodb(mongo_uri_ui or mongo_uri_env, mongo_db, mongo_coll, query=q)
            except Exception as e:
                st.error(f"MongoDB ingestion failed: {e}")

    elif source_choice == "Manual Entry":
        if not st.session_state.manual_saved:
            df = manual_log_entry()
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.raw_df = df
                st.session_state.manual_saved = True
                st.session_state.current_log = 1
                st.session_state.logs = []
                safe_rerun()
        else:
            df = st.session_state.df
        print(df.dtypes)
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
                        p = preprocess_df_keepall(df, text_cols)
                        st.session_state['processed'] = p
                        st.success('Preprocessing complete')
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")
                        st.stop()
            
                    try:
                        embeddings = embed_texts(p['clean_text'].tolist())
                        st.session_state['embeddings'] = embeddings
                        st.success('Embeddings computed')
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")
            # --- Only show analysis after preprocessing & embeddings ---
            if 'processed' in st.session_state and 'embeddings' in st.session_state:
                p = st.session_state.get('processed')
                embeddings = st.session_state.get('embeddings')
            
                # Guard checks
                valid_p = isinstance(p, pd.DataFrame) and not p.empty
                valid_embeddings = embeddings is not None and len(embeddings) > 0
        
            # --- Clustering ---
            st.subheader("Clustering & Visualization")
        
            if valid_p and valid_embeddings:
                if st.button('Cluster & Visualize'):
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
        
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")
        
            if 'cluster_fig' in st.session_state:
                st.success(st.session_state['cluster_text'])
                st.info(st.session_state['cluster_metrics']["interpretation"])
                st.plotly_chart(st.session_state['cluster_fig'], use_container_width=True)
            else:
                st.warning("Processed data or embeddings are not available. Please run Preprocess & Embed first.")

            
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
            
            
            # --- Pareto Analysis ---
            st.subheader("Pareto Analysis")
            p = st.session_state.get('processed')
            
            def pareto_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
                if column not in df.columns:
                    return pd.DataFrame()
                if pd.api.types.is_numeric_dtype(df[column]):
                    series = df[column].dropna().astype(str)
                else:
                    series = df[column].dropna().astype(str).str.strip()
                series = series[series != ""]
                if series.empty:
                    return pd.DataFrame()
                counts = series.value_counts()
                total = counts.sum()
                tab = pd.DataFrame({
                    "Category": counts.index,
                    "Count": counts.values,
                })
                tab["Percent"] = (tab["Count"] / total * 100).round(2)
                tab["Cumulative %"] = tab["Percent"].cumsum().round(2)
                return tab
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    cat_col = st.selectbox(
                        'Select column for Pareto',
                        options=p.columns.tolist(),
                        help="Choose any column from processed data"
                    )
                    if st.button('Show Pareto'):
                        st.session_state['show_pareto'] = True
                        st.session_state['pareto_col'] = cat_col
                    if st.session_state.get('show_pareto', False):
                        try:
                            selected_col = st.session_state.get('pareto_col', cat_col)
                            tab = pareto_table(p, selected_col)
                            if tab.empty:
                                st.warning(f"No valid data found in column '{selected_col}'.")
                            else:
                                fig = pareto_plot(tab)
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Pareto failed: {e}")
                except Exception as e:
                    st.error(f"Pareto setup failed: {e}")
            else:
                st.warning("No processed data available for Pareto analysis. Please preprocess first.")
            
            
            # --- SPC Section ---
            st.subheader("Statistical Process Control (SPC)")
            p = st.session_state.get('processed')
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    spc_df = p.copy()
                    for c in spc_df.columns:
                        if not pd.api.types.is_numeric_dtype(spc_df[c]):
                            spc_df[c] = pd.to_numeric(spc_df[c], errors='coerce')
                    num_cols = [c for c in spc_df.select_dtypes(include=['number']).columns if spc_df[c].notna().any()]
                    if num_cols:
                        spc_col_selected = st.selectbox('Select numeric column for SPC', options=num_cols, key='spc_col_select')
                        subgroup_size = st.number_input('Subgroup Size (1 = I-MR chart)', min_value=1, value=1, key='spc_subgroup')
                        time_cols = [c for c in spc_df.columns if pd.api.types.is_datetime64_any_dtype(spc_df[c])]
                        for c in spc_df.select_dtypes(include=['object']).columns:
                            try:
                                spc_df[c] = pd.to_datetime(spc_df[c], errors='coerce')
                                if spc_df[c].notna().any():
                                    time_cols.append(c)
                            except Exception:
                                continue
                        time_col_selected = st.selectbox('Optional time column', options=[None] + time_cols, key='spc_time_col_select')
                        if st.button('Show SPC Chart', key='spc_btn'):
                            try:
                                from visualization import plot_spc_chart
                                fig_spc = plot_spc_chart(spc_df, spc_col_selected, subgroup_size=subgroup_size, time_col=time_col_selected)
                                st.session_state['spc_fig'] = fig_spc
                                st.session_state['spc_col_saved'] = spc_col_selected
                            except Exception as e:
                                st.error(f"SPC plotting failed: {e}")
                        if 'spc_fig' in st.session_state:
                            st.success(f"SPC Chart for: {st.session_state.get('spc_col_saved', '')}")
                            st.plotly_chart(
                                st.session_state['spc_fig'],
                                use_container_width=True,
                                key=f"spc_chart_{st.session_state.get('spc_col_saved', '')}"
                            )
                    else:
                        st.info("No numeric columns available for SPC analysis after conversion.")
                except Exception as e:
                    st.error(f"SPC setup failed: {e}")
            else:
                st.warning("No processed data available for SPC. Please preprocess first.")
            
            
            # --- Trend Dashboard ---
            # --- Trend Dashboard ---
            st.subheader("📈 Trend Dashboard")
            
            p = st.session_state.get("processed")
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    trend_df = p.copy()
            
                    # --- Detect numeric columns ---
                    for c in trend_df.columns:
                        if trend_df[c].dtype == "object":
                            trend_df[c] = pd.to_numeric(
                                trend_df[c].astype(str).str.replace(",", "").str.strip(),
                                errors="ignore"
                            )
            
                    num_cols = [c for c in trend_df.select_dtypes(include=['number']).columns if trend_df[c].notna().any()]
            
                    # --- Detect date columns (with optional format) ---
                    date_cols = []
                    for c in trend_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(trend_df[c]):
                            date_cols.append(c)
                        else:
                            try:
                                if st.session_state.get("date_format"):  # only use if defined
                                    converted = pd.to_datetime(
                                        trend_df[c].astype(str).str.strip(),
                                        format=st.session_state["date_format"],
                                        errors="coerce"
                                    )
                                else:
                                    converted = pd.to_datetime(trend_df[c].astype(str).str.strip(), errors="coerce")
            
                                if converted.notna().any():
                                    trend_df[c] = converted
                                    if c not in date_cols:
                                        date_cols.append(c)
                            except Exception:
                                continue
            
                    # --- Build Dashboard ---
                    if date_cols and num_cols:
                        date_col = st.selectbox("Select Date Column", options=date_cols, key="trend_date_col")
                        value_col = st.selectbox("Select Value Column", options=num_cols, key="trend_value_col")
            
                        if st.button("Show Dashboard", key="trend_btn"):
                            st.session_state['show_trend'] = True
                            st.session_state['trend_date'] = date_col
                            st.session_state['trend_value'] = value_col
            
                        if st.session_state.get('show_trend', False):
                            try:
                                fig_trend = plot_trend_dashboard(
                                    trend_df,
                                    date_col=st.session_state.get('trend_date', date_col),
                                    value_col=st.session_state.get('trend_value', value_col),
                                )
                                if fig_trend:
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                else:
                                    st.warning("⚠️ Selected columns are invalid for plotting.")
                            except Exception:
                                st.info("⚠️ Unable to render trend plot. Please check your data.")
                    else:
                        st.info("No valid date and numeric column pair available for trend plotting.")
            
                except Exception:
                    st.info("⚠️ Trend Dashboard could not be built for this dataset.")
            else:
                st.info("No processed data available for Trend Dashboard. Please preprocess first.")

            
            
            # --- Time-Series Trend Analysis ---
            st.subheader("⏳ Time-Series Trend Analysis")
            if isinstance(p, pd.DataFrame) and not p.empty:
                if date_cols and num_cols:
                    time_col = st.selectbox("Select time column", options=date_cols, key="time_col")
                    value_col = st.selectbox("Select value column", options=num_cols, key="time_value_col")
                    freq_options = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
                    freq_choice = st.selectbox("Select aggregation level", options=list(freq_options.keys()))
                    agg_options = ["mean", "sum", "max", "min"]
                    agg_choice = st.selectbox("Select aggregation function", options=agg_options)
                    if st.button("Plot Time-Series Trend", key="time_btn"):
                        # Apply global format to selected column
                        if st.session_state["date_format"]:
                            p[time_col] = pd.to_datetime(p[time_col].astype(str).str.strip(), format=st.session_state["date_format"], errors="coerce")
                        else:
                            p[time_col] = pd.to_datetime(p[time_col].astype(str).str.strip(), errors="coerce")
                        fig_time = plot_time_series_trend(
                            p, time_col, value_col,
                            freq=freq_options[freq_choice],
                            agg_func=agg_choice
                        )
                        if fig_time:
                            st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.warning("No valid datetime and numeric column pair for time-series analysis.")
            else:
                st.warning("No processed data available. Please preprocess first.")
            
            

            # --- Root Cause Analysis (RCA) ---
    
            st.subheader("Root Cause Analysis (RCA)")
            
            p = st.session_state.get("processed")
            
            if isinstance(p, pd.DataFrame) and not p.empty:
                try:
                    # Row selector
                    idx = st.number_input(
                        "Pick row index to analyze",
                        min_value=0,
                        max_value=len(p) - 1,
                        value=0,
                    )
                    row = p.iloc[int(idx)]
            
                    # Preview text
                    st.markdown("**Selected row preview:**")
                    text_preview = row.get("combined_text") or row.get("clean_text") or ""
                    st.write(text_preview)
            
                    # RCA mode selector
                    mode = st.radio("RCA Mode", options=["AI-Powered (LLM)", "Rule-Based (fallback)"])
            
                    # RCA execution
                    if st.button("Run RCA"):
                        with st.spinner("Running RCA..."):
                            try:
                                if mode == "AI-Powered (LLM)":
                                    st.session_state["rca_result"] = ai_rca_with_fallback(
                                        str(row.get("combined_text", "")),
                                        str(row.get("clean_text", "")),
                                    )
                                else:
                                    fb = rule_based_rca_suggestions(str(row.get("clean_text", "")))
                                    st.session_state["rca_result"] = {
                                        "from": "rule_based",
                                        "fishbone": fb,
                                    }
                            except Exception as e:
                                st.session_state["rca_result"] = {"error": f"RCA failed: {e}"}
            
                    # Display results
                    result = st.session_state.get("rca_result", {})
            
                    if result:
                        col1, col2 = st.columns([1, 1])
            
                        # --- Left column: RCA details ---
                        with col1:
                            st.markdown("### RCA - Details")
            
                            if result.get("error"):
                                st.error(result.get("error"))
            
                            if result.get("root_causes"):
                                st.markdown("**Root causes:**")
                                st.json(result.get("root_causes"))
            
                            if result.get("five_whys"):
                                st.markdown("**5-Whys**")
                                for i, w in enumerate(result.get("five_whys"), start=1):
                                    st.write(f"{i}. {w}")
            
                            if result.get("capa"):
                                st.markdown("**CAPA Recommendations**")
                                for capa in result.get("capa"):
                                    st.write(
                                        f"- **{capa.get('type', '')}**: {capa.get('action', '')} "
                                        f"(Owner: {capa.get('owner', '')}, due in {capa.get('due_in_days', '?')} days)"
                                    )
            
                            if result.get("fishbone") and not result.get("root_causes"):
                                st.markdown("**Fishbone (rule-based)**")
                                st.json(result.get("fishbone"))
            
                        # --- Right column: Fishbone diagram ---
                        with col2:
                            st.markdown("### Fishbone Diagram")
                            fishbone_data = result.get("fishbone") or {}
            
                            # If fishbone is empty, try to build from root causes
                            if not fishbone_data:
                                fishbone_data = {
                                    k: [] for k in ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
                                }
                                for rc in (result.get("root_causes") or []):
                                    if isinstance(rc, dict):
                                        cat = rc.get("category") or "Method"
                                        fishbone_data.setdefault(cat, []).append(rc.get("cause") or "")
            
                            # Render fishbone if any data exists
                            if not any(fishbone_data.values()):
                                st.info("No fishbone data available to plot.")
                            else:
                                try:
                                    fig = visualize_fishbone(fishbone_data)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Fishbone visualization failed: {e}")
                                    st.json(fishbone_data)
            
                except Exception as e:
                    st.error(f"RCA setup failed: {e}")
            
            else:
                st.warning("No processed data available for RCA. Please preprocess first.")

                # --- Manual 5-Whys & CAPA creation ---
                st.markdown("---")
                st.subheader("Manual 5-Whys & CAPA creation")
                
                manual_whys = []
                for i in range(5):
                    manual_whys.append(st.text_input(f"Why {i+1}", key=f"manual_why_{i}"))
                
                with st.form("capa_form"):
                    labels = st.session_state.get('labels', [])
                    label_count = len(labels) if isinstance(labels, list) else 0
                    default_issue_id = f"issue-{label_count}-{st.session_state.get('current_log', 1)}"
                    
                    desc_default = ""
                    p = st.session_state.get('processed')
                    if isinstance(p, pd.DataFrame) and not p.empty:
                        try:
                            desc_default = str(p.iloc[0].get("combined_text", p.iloc[0].get("clean_text", "")))
                        except Exception:
                            desc_default = ""
                
                    issue_id = st.text_input("Issue ID", value=default_issue_id)
                    desc = st.text_area("Description", value=desc_default)
                    corrective = st.text_area("Corrective Action")
                    preventive = st.text_area("Preventive Action")
                    owner = st.text_input("Owner")
                    due_days = st.number_input("Due in (days)", min_value=1, max_value=365, value=14)
                    submitted = st.form_submit_button("Create CAPA")
                    if submitted:
                        try:
                            db = SessionLocal()
                            from datetime import datetime, timedelta
                            due_date = datetime.utcnow() + timedelta(days=int(due_days))
                            capa = CAPA(
                                issue_id=issue_id,
                                description=desc,
                                corrective_action=(corrective + "\n\nPreventive:\n" + preventive),
                                owner=owner,
                                due_date=due_date,
                            )
                            db.add(capa)
                            db.commit()
                            st.success("CAPA created and saved to DB")
                        except Exception as e:
                            st.error(f"Failed to save CAPA: {e}")


if __name__ == "__main__":
    main()
