# ================================
# File: src/streamlit_app.py
# ================================
import os
import sys
import streamlit as st
import pandas as pd
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
from clustering import fit_kmeans
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

    df = raw_df.drop(index=row_idx).copy()
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
            except Exception:
                pass

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

    # ----------------- Data Preview and downstream workflow -----------------
 
    # Ensure DataFrame from manual logs is captured
    if df is None and st.session_state.get("manual_df_ready"):
        df = st.session_state.df
    
    if df is not None:
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.session_state.raw_df = df
            st.session_state.header_row = 0
    
            # Only apply row as header if NOT from manual logs
            if not st.session_state.get("manual_df_ready"):
                st.session_state.df = apply_row_as_header(df, 0)
            else:
                st.session_state.df = df  # use as-is for manual entry
    
            st.success(f"Data loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns.")
        else:
            st.warning("Ingested data is empty or not a DataFrame.")


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
            # Preprocess & embed
            st.markdown("### Text Selection")
            object_cols = [c for c in df.columns if df[c].dtype == 'object']
            default_text_cols = object_cols[:2]
            text_cols = st.multiselect(
                'Text columns to use for embedding',
                options=df.columns.tolist(),
                default=default_text_cols
            )

            if st.button('Preprocess & Embed'):
                if not text_cols:
                    st.error("Please select at least one text column.")
                else:
                    try:
                        p = preprocess_df(df, text_cols)
                        st.session_state['processed'] = p
                        st.success('Preprocessing complete')
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")
                        return

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
        
                # Guard that p is a non-empty DataFrame and embeddings are available
                valid_p = isinstance(p, pd.DataFrame) and not p.empty
                valid_embeddings = embeddings is not None and len(embeddings) > 0
        
                # --- Clustering ---
                st.subheader("Clustering & Visualization")
                if valid_p and valid_embeddings:
                    if st.button('Cluster & Visualize'):
                        try:
                            km, labels, score, interpretation = fit_kmeans(embeddings)
                            st.write(f"Silhouette score: {score:.3f}")
                            if interpretation:
                                st.info(interpretation)
                            st.session_state['labels'] = labels
                            fig = cluster_scatter(embeddings, labels)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Clustering failed: {e}")
                else:
                    st.warning("Processed data or embeddings are not available. Please run Preprocess & Embed first.")
        
               # --- Pareto Analysis ---
                st.subheader("Pareto Analysis")
                p = st.session_state.get('processed')  # re-fetch to be safe after any rerun
                
                if isinstance(p, pd.DataFrame) and not p.empty:
                    try:
                        cat_col = st.selectbox(
                            'Select column for Pareto',
                            options=p.columns.tolist()
                        )
                        if st.button('Show Pareto'):
                            try:
                                tab = pareto_table(p, cat_col)
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
                        num_cols = p.select_dtypes(include=['number']).columns.tolist()
                        if num_cols:
                            spc_col = st.selectbox('Select numeric column for SPC', options=num_cols)
                            if st.button('Show SPC Chart'):
                                try:
                                    fig_spc = plot_spc_chart(p, spc_col)
                                    st.plotly_chart(fig_spc, use_container_width=True)
                                except Exception as e:
                                    st.error(f"SPC chart failed: {e}")
                        else:
                            st.info("No numeric columns available for SPC analysis.")
                    except Exception as e:
                        st.error(f"SPC setup failed: {e}")
                else:
                    st.warning("No processed data available for SPC. Please preprocess first.")


                # --- Trend Dashboard ---
                st.subheader("Trend Dashboard")
                p = st.session_state.get('processed')
                if isinstance(p, pd.DataFrame) and not p.empty:
                    if len(p.columns) >= 2:
                        # Let user select which columns to use
                        date_col = st.selectbox("Select Date Column", options=p.columns)
                        value_col = st.selectbox("Select Value Column", options=p.columns)
                
                        if st.button("Show Dashboard"):
                            try:
                                fig_trend = plot_trend_dashboard(p, date_col=date_col, value_col=value_col)
                                if fig_trend:
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                else:
                                    st.warning("Selected columns are invalid for plotting.")
                            except Exception as e:
                                st.error(f"Trend dashboard failed: {e}")
                    else:
                        st.warning("Not enough columns to plot a trend dashboard. Need at least 2 columns.")
                else:
                    st.warning("No processed data available for Trend Dashboard. Please preprocess first.")

                # --- Time-Series Trend Analysis ---
                st.subheader("Time-Series Trend Analysis")
                p = st.session_state.get('processed')
                if isinstance(p, pd.DataFrame) and not p.empty:
                    try:
                        time_cols = [c for c in p.columns if pd.api.types.is_datetime64_any_dtype(p[c])]
                        if time_cols:
                            time_col = st.selectbox("Select time column for trend analysis", options=time_cols)
                            numeric_cols = p.select_dtypes(include=['number']).columns.tolist()
                            if numeric_cols:
                                value_col = st.selectbox("Select value column for trend", options=numeric_cols)
                                if st.button("Plot Time-Series Trend"):
                                    try:
                                        fig_time = plot_time_series_trend(p, time_col, value_col)
                                        st.plotly_chart(fig_time, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Time-series trend failed: {e}")
                            else:
                                st.info("No numeric columns available for time-series value selection.")
                        else:
                            st.info("No datetime column detected for time-series analysis.")
                    except Exception as e:
                        st.error(f"Time-series setup failed: {e}")
                else:
                    st.warning("No processed data available for Time-Series analysis. Please preprocess first.")
        
                # --- Root Cause Analysis (RCA) ---
                st.subheader("Root Cause Analysis (RCA)")
                p = st.session_state.get('processed')
                if isinstance(p, pd.DataFrame) and not p.empty:
                    try:
                        idx = st.number_input('Pick row index to analyze', min_value=0, max_value=len(p)-1, value=0)
                        row = p.iloc[int(idx)]
                        st.markdown("**Selected row preview:**")
                        st.write(row.get('combined_text', row.get('clean_text', '')))
        
                        mode = st.radio("RCA Mode", options=["AI-Powered (LLM)", "Rule-Based (fallback)"])
        
                        result = {}
                        if st.button("Run RCA"):
                            with st.spinner("Running RCA..."):
                                try:
                                    if mode == "AI-Powered (LLM)":
                                        result = ai_rca_with_fallback(
                                            str(row.get('combined_text', '')),
                                            str(row.get('clean_text', ''))
                                        )
                                    else:
                                        fb = rule_based_rca_suggestions(str(row.get('clean_text', '')))
                                        result = {"from": "rule_based", "fishbone": fb}
                                except Exception as e:
                                    result = {"error": f"RCA failed: {e}"}
        
                        if result:
                            col1, col2 = st.columns([1, 1])
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
                                        st.write(f"- **{capa.get('type', '')}**: {capa.get('action', '')} "
                                                 f"(Owner: {capa.get('owner', '')}, due in {capa.get('due_in_days', '?')} days)")
                                if result.get("fishbone") and not result.get("root_causes"):
                                    st.markdown("**Fishbone (rule-based)**")
                                    st.json(result.get("fishbone"))
        
                            with col2:
                                st.markdown("### Fishbone Diagram")
                                fishbone_data = result.get("fishbone") or {}
                                if not fishbone_data:
                                    # build from root causes if provided
                                    fishbone_data = {k: [] for k in ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]}
                                    for rc in (result.get("root_causes") or []):
                                        if isinstance(rc, dict):
                                            cat = rc.get("category") or "Method"
                                            fishbone_data.setdefault(cat, []).append(rc.get("cause") or "")
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
