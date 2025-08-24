# ================================
# File: src/streamlit_app.py
# ================================
import os
import sys
import streamlit as st
import pandas as pd

# -----------------------------
# Ensure import paths are correct
# -----------------------------
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
for p in [FILE_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Local package imports
from ingestion import ingest_file, manual_log_entry, save_processed
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


# Safe rerun utility (kept for compatibility; not needed for the header change)
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ---------- Helpers for the new "Row as Header" feature ----------
def _make_unique(names):
    """Ensure column names are unique and non-empty."""
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
    """Return a new DataFrame whose columns come from the given row index of raw_df."""
    if raw_df is None or raw_df.empty:
        return raw_df
    row_idx = int(max(0, min(row_idx, len(raw_df) - 1)))
    new_header = raw_df.iloc[row_idx].astype(str).tolist()
    new_header = _make_unique(new_header)

    # Keep all rows intact, only change column names
    df = raw_df.copy()
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)
    return df


    # Normalize types a bit (keep strings as strings; try parsing date-like columns)
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
        # leave other columns as-is; preprocessing handles text later
    return df


def main():
    st.set_page_config(page_title='Smart NC Analyzer', layout='wide')
    st.title('Smart Non-Conformance Analyzer')

    # Initialize database
    try:
        init_db()
    except Exception as e:
        st.warning(f"Database init warning: {e}")

    # Sidebar upload
    st.sidebar.header('Upload')
    uploaded = st.sidebar.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])

    # Sidebar data input
    st.sidebar.header("Data Input Method")
    options = ["Upload File"]
    manual_entry_disabled = uploaded is not None
    if not manual_entry_disabled:
        options.append("Manual Entry")
    else:
        st.sidebar.info("Close the uploaded file to continue in Manual Entry Mode.")

    source_choice = st.sidebar.radio("Select Input Method", options)

    # ---------- Session state ----------
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None     # the original uploaded/manual dataframe
    if "df" not in st.session_state:
        st.session_state.df = None         # the working df (after applying header row)
    if "header_row" not in st.session_state:
        st.session_state.header_row = 0    # selected row index to use as header
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1

    # ---------- Ingestion ----------
    if source_choice == "Upload File":
        if uploaded is None:
            st.session_state.raw_df = None
            st.session_state.df = None
        else:
            df = ingest_file(uploaded)
            if df is not None and not df.empty:
                st.session_state.raw_df = df
                # reset header selection for a new file
                st.session_state.header_row = 0
                # immediately apply row-as-header to produce working df
                st.session_state.df = apply_row_as_header(st.session_state.raw_df, st.session_state.header_row)
                try:
                    # Try caching the raw df as-is first
                    save_processed(df, "uploaded_data.parquet")  # cache original upload (raw)
                except Exception as e:
                    # Fallback: make object columns explicit strings just for caching
                    try:
                        df_cache = df.copy()
                        for c in df_cache.columns:
                            if df_cache[c].dtype == "object":
                                df_cache[c] = df_cache[c].astype(str)
                        save_processed(df_cache, "uploaded_data.parquet")
                    except Exception as e2:
                        st.info(f"Could not cache uploaded data: {e2}")
            else:
                st.warning("Uploaded file is empty or invalid.")
                st.session_state.raw_df = None
                st.session_state.df = None

    elif source_choice == "Manual Entry":
        df = manual_log_entry()
        if df is not None and not df.empty:
            st.session_state.raw_df = df
            st.session_state.header_row = 0
            st.session_state.df = apply_row_as_header(st.session_state.raw_df, st.session_state.header_row)
            try:
                # Try caching the raw df as-is first
                save_processed(df, "manual_data.parquet")  # cache original manual entry (raw)
            except Exception as e:
                # Fallback: make object columns explicit strings just for caching
                try:
                    df_cache = df.copy()
                    for c in df_cache.columns:
                        if df_cache[c].dtype == "object":
                            df_cache[c] = df_cache[c].astype(str)
                    save_processed(df_cache, "manual_data.parquet")
                except Exception as e2:
                    st.info(f"Could not cache manual data: {e2}")
        else:
            st.session_state.raw_df = None
            st.session_state.df = None

    # ---------- UI: Data preview + NEW header selector ----------
    if st.session_state.raw_df is not None and not st.session_state.raw_df.empty:
        st.subheader("Raw Data Preview (after applying selected header row)")

        # Header row selector (IMMEDIATE EFFECT: changes df on widget change)
        max_row = len(st.session_state.raw_df) - 1
        new_header_row = st.number_input(
            "Row number to use as header (0-indexed)",
            min_value=0, max_value=max_row, value=int(st.session_state.header_row), step=1,
            help="Choose a row from the original upload to become the column headers. "
                 "The preview and the text column picker will update immediately."
        )

        # If user changed the row, re-apply header against the ORIGINAL raw_df
        if int(new_header_row) != int(st.session_state.header_row):
            st.session_state.header_row = int(new_header_row)
            st.session_state.df = apply_row_as_header(st.session_state.raw_df, st.session_state.header_row)
            # No manual rerun needed; Streamlit reruns automatically on widget change

        # Show preview of the working df (after header application)
        df = st.session_state.df.copy()
        df_display = (
            df.reset_index(drop=True)
              .rename_axis("No")
              .rename(lambda x: x + 1, axis=0)  # start index from 1
        )
        st.dataframe(df_display.head(50))

        # ---------- Preprocess & Embed ----------
        st.markdown("### Text Selection")
        # Use UPDATED columns (post header-application) to determine object/text-like columns
        object_cols = [c for c in df.columns if df[c].dtype == 'object']
        default_text_cols = object_cols[:2]  # you can change to object_cols to default-select all text columns
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

        # --- Only show clustering, Pareto, SPC after preprocessing ---
        if 'processed' in st.session_state and 'embeddings' in st.session_state:
            p = st.session_state['processed']
            embeddings = st.session_state['embeddings']

            # --- Clustering ---
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

            # --- Pareto Analysis ---
            st.subheader("Pareto Analysis")
            cat_col = st.selectbox('Select column for Pareto', options=p.columns.tolist())
            if st.button('Show Pareto'):
                try:
                    tab = pareto_table(p, cat_col)
                    fig = pareto_plot(tab)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Pareto failed: {e}")

            # --- SPC Section ---
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

            # --- Trend Dashboard ---
            st.subheader("Trend Dashboard")
            if st.button("Show Dashboard"):
                try:
                    fig_trend = plot_trend_dashboard(p)
                    st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.error(f"Trend dashboard failed: {e}")

            # --- Time-Series Trend Analysis ---
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

            # --- Root Cause Analysis (RCA) ---
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

            # --- Manual 5-Whys & CAPA creation ---
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


if __name__ == "__main__":
    main()
