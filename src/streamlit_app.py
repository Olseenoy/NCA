# ================================
# File: src/streamlit_app.py
# ================================
import os
import sys
import streamlit as st
import pandas as pd
from typing import Optional, Dict
from dotenv import load_dotenv, set_key, find_dotenv
import json

# -----------------------------
# Ensure import paths are correct
# -----------------------------
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
for p in [FILE_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Local package imports
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

    # ---------------- Session defaults ----------------
    defaults = ["raw_df", "df", "header_row", "logs", "current_log", "manual_saved", "processed", "embeddings", "labels", "creds", "source_choice"]
    for key in defaults:
        if key not in st.session_state:
            if key == "current_log":
                st.session_state[key] = 1
            elif key == "manual_saved":
                st.session_state[key] = False
            elif key == "creds":
                st.session_state[key] = {}
            else:
                st.session_state[key] = None

    # ---------------- Sidebar: Source + Auth settings ----------------
    st.sidebar.header("Data Input")
    options = [
        "Upload File (CSV/Excel)",
        "Google Sheets",
        "OneDrive / SharePoint",
        "REST API (ERP/MES/QMS)",
        "SQL Database",
        "MongoDB",
        "Manual Entry",
    ]
    new_choice = st.sidebar.selectbox(
        "Select input method",
        options,
        index=0 if st.session_state.source_choice is None else options.index(st.session_state.source_choice)
    )

    # Detect source switch
    if st.session_state.source_choice and new_choice != st.session_state.source_choice:
        st.warning("Switching data source will end the current session and clear all data.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, end session and switch"):
                keys_to_clear = ["raw_df","df","header_row","logs","current_log","manual_saved","processed","embeddings","labels"]
                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.source_choice = new_choice
                safe_rerun()
        with col2:
            if st.button("No, keep current session"):
                new_choice = st.session_state.source_choice

    st.session_state.source_choice = new_choice
    source_choice = new_choice

    # ---------------- Sidebar: Auth ----------------
    with st.sidebar.expander("🔒 Authentication & Credentials (expand to override)"):
        st.markdown("Credentials loaded from environment variables by default.")
        cred_inputs = {}
        for k, label in CRED_KEYS.items():
            is_secret = "SECRET" in k or "TOKEN" in k or "PASSWORD" in k
            default = get_cred_value(k)
            cred_inputs[k] = st.text_input(label, value=default, key=f"cred_{k}", type="password" if is_secret else "default")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save for Session Only"):
                session_pairs = {k:v for k,v in cred_inputs.items() if v}
                save_creds_to_session(session_pairs)
        with col2:
            if st.button("Save to .env Permanently"):
                env_pairs = {k:v for k,v in cred_inputs.items() if v}
                try:
                    save_creds_to_env(env_pairs)
                except Exception as e:
                    st.error(f"Failed to write to .env: {e}")

    # ----------------- Ingestion logic -----------------
    df = None

    if source_choice == "Upload File (CSV/Excel)":
        uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv','xlsx','xls'])
        if uploaded:
            try: df = ingest_file(uploaded)
            except Exception as e: st.error(f"File ingestion failed: {e}")

    elif source_choice == "Google Sheets":
        sheet_url = st.sidebar.text_input("Sheet URL or ID")
        sa_path = get_cred_value("GOOGLE_SERVICE_ACCOUNT_JSON")
        api_key = get_cred_value("GOOGLE_API_KEY")
        use_service_account = st.sidebar.checkbox("Use service account JSON", value=bool(sa_path))
        if st.sidebar.button("Load Google Sheet"):
            try: df = ingest_google_sheet(sheet_url, service_account_json_path=sa_path if use_service_account else None, api_key=api_key)
            except Exception as e: st.error(f"Google Sheets ingestion failed: {e}")

    elif source_choice == "OneDrive / SharePoint":
        od_file = st.sidebar.text_input("File path / URL")
        if st.sidebar.button("Load from OneDrive"):
            try: df = ingest_onedrive(od_file)
            except Exception as e: st.error(f"OneDrive ingestion failed: {e}")

    elif source_choice == "REST API (ERP/MES/QMS)":
        api_url = st.sidebar.text_input("API URL")
        if st.sidebar.button("Fetch API"):
            try: df = ingest_rest_api(api_url)
            except Exception as e: st.error(f"REST API ingestion failed: {e}")

    elif source_choice == "SQL Database":
        sql_query = st.sidebar.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 100")
        db_conn = get_cred_value("DB_CONN")
        if st.sidebar.button("Run Query"):
            try: df = ingest_database(db_conn, sql_query)
            except Exception as e: st.error(f"Database ingestion failed: {e}")

    elif source_choice == "MongoDB":
        mongo_uri = get_cred_value("MONGO_URI")
        if st.sidebar.button("Load MongoDB"):
            try: df = ingest_mongodb(mongo_uri, "db_name","collection_name")
            except Exception as e: st.error(f"MongoDB ingestion failed: {e}")

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

    # ----------------- Data Preview & downstream workflow -----------------
    if df is not None:
        if isinstance(df,pd.DataFrame) and not df.empty:
            st.session_state.raw_df = df
            st.session_state.header_row = 0
            st.session_state.df = apply_row_as_header(df, 0)
            st.success(f"Data loaded: {len(df)} rows, {len(df.columns)} columns.")
        else:
            st.warning("Ingested data is empty or not a DataFrame.")

    # Main area: only show preview/analysis if raw_df present
    if st.session_state.get("raw_df") is not None and not st.session_state.get("raw_df").empty:
        st.subheader("Data Preview & Actions")

        df = st.session_state.df

        # Header selector for uploaded/raw data
        max_row = len(st.session_state.raw_df) - 1
        new_header_row = st.number_input(
            "Row number to use as header (0-indexed)",
            min_value=0, max_value=max_row,
            value=int(st.session_state.header_row) if st.session_state.header_row is not None else 0,
            step=1,
            help="Pick a row from the file to become column headers."
        )
        if int(new_header_row) != int(st.session_state.header_row):
            st.session_state.header_row = int(new_header_row)
            st.session_state.df = apply_row_as_header(st.session_state.raw_df, st.session_state.header_row)
            df = st.session_state.df
            st.experimental_rerun()

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

            # Only show analytics if processed + embeddings present
            if 'processed' in st.session_state and 'embeddings' in st.session_state:
                p = st.session_state['processed']
                embeddings = st.session_state['embeddings']

                # Clustering
                st.subheader("Clustering & Visualization")
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

                # Pareto
                st.subheader("Pareto Analysis")
                cat_col = st.selectbox('Select column for Pareto', options=p.columns.tolist())
                if st.button('Show Pareto'):
                    try:
                        tab = pareto_table(p, cat_col)
                        fig = pareto_plot(tab)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Pareto failed: {e}")

                # SPC
                st.subheader("Statistical Process Control (SPC)")
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

                # Trend Dashboard
                st.subheader("Trend Dashboard")
                if st.button("Show Dashboard"):
                    try:
                        fig_trend = plot_trend_dashboard(p)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    except Exception as e:
                        st.error(f"Trend dashboard failed: {e}")

                # Time-Series Trend
                st.subheader("Time-Series Trend Analysis")
                time_cols = [c for c in p.columns if pd.api.types.is_datetime64_any_dtype(p[c])]
                if time_cols:
                    time_col = st.selectbox("Select time column for trend analysis", options=time_cols)
                    value_col = st.selectbox(
                        "Select value column for trend",
                        options=p.select_dtypes(include=['number']).columns.tolist()
                    )
                    if st.button("Plot Time-Series Trend"):
                        try:
                            fig_time = plot_time_series_trend(p, time_col, value_col)
                            st.plotly_chart(fig_time, use_container_width=True)
                        except Exception as e:
                            st.error(f"Time-series trend failed: {e}")
                else:
                    st.info("No datetime column detected for time-series analysis.")

                # RCA
                st.subheader("Root Cause Analysis (RCA)")
                if len(p) == 0:
                    st.info("No rows to analyze.")
                else:
                    idx = st.number_input('Pick row index to analyze', min_value=0, max_value=len(p)-1, value=0)
                    row = p.iloc[int(idx)]
                    st.markdown("**Selected row preview:**")
                    st.write(row.get('combined_text', row.get('clean_text', '')))

                    mode = st.radio("RCA Mode", options=["AI-Powered (LLM)", "Rule-Based (fallback)"])

                    if st.button("Run RCA"):
                        with st.spinner("Running RCA..."):
                            try:
                                if mode == "AI-Powered (LLM)":
                                    result = ai_rca_with_fallback(str(row.get('combined_text', '')), str(row.get('clean_text', '')))
                                else:
                                    fb = rule_based_rca_suggestions(str(row.get('clean_text', '')))
                                    result = {"from": "rule_based", "fishbone": fb}
                            except Exception as e:
                                result = {"error": f"RCA failed: {e}"}

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("### RCA - Details")
                            if result.get("error"):
                                st.error(result.get("error"))
                            if result.get("root_causes"):
                                st.markdown("**Root causes:**")
                                st.json(result.get("root_causes"))
                            if result.get("five_whys"):
                                st.markdown("**5-Whys")
                                for i, w in enumerate(result.get("five_whys"), start=1):
                                    st.write(f"{i}. {w}")
                            if result.get("capa"):
                                st.markdown("**CAPA Recommendations**")
                                for capa in result.get("capa"):
                                    st.write(f"- **{capa.get('type', '')}**: {capa.get('action', '')} (Owner: {capa.get('owner', '')}, due in {capa.get('due_in_days', '?')} days)")
                            if result.get("fishbone") and not result.get("root_causes"):
                                st.markdown("**Fishbone (rule-based)**")
                                st.json(result.get("fishbone"))

                        with col2:
                            st.markdown("### Fishbone Diagram")
                            fishbone_data = result.get("fishbone") or {}
                            if not fishbone_data:
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

                # Manual 5-Whys & CAPA creation
                st.markdown("---")
                st.subheader("Manual 5-Whys & CAPA creation")
                manual_whys = []
                for i in range(5):
                    manual_whys.append(st.text_input(f"Why {i+1}", key=f"manual_why_{i}"))

                with st.form("capa_form"):
                    default_issue_id = f"issue-{len(st.session_state.get('labels', [])) or 0}-{st.session_state.get('current_log', 1)}"
                    issue_id = st.text_input("Issue ID", value=default_issue_id)
                    desc = st.text_area("Description", value=str(p.iloc[0].get("combined_text", "")) if len(p) else "")
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

    else:
        st.info("No data loaded. Use the sidebar to upload or connect to a data source.")

if __name__ == "__main__":
    main()
