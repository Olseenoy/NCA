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


# ---------------- Safe Rerun ----------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ----------------- Helpers -----------------
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
    """
    Apply a row from raw_df as header and return updated DataFrame.
    Row numbering starts from 0 (first row of uploaded file).
    """
    if raw_df is None or raw_df.empty:
        return raw_df

    row_idx = int(max(0, min(row_idx, len(raw_df) - 1)))

    # Extract header values
    new_header = raw_df.iloc[row_idx].astype(str).tolist()
    new_header = _make_unique(new_header)

    # Drop header row from data
    df = raw_df.drop(index=row_idx).copy()
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    # Try parsing dates for columns with "date" in name
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
            except Exception:
                pass

    return df


# ----------------- Main App -----------------
def main():
    st.set_page_config(page_title='Smart NC Analyzer', layout='wide')
    st.title('Smart Non-Conformance Analyzer')

    try:
        init_db()
    except Exception as e:
        st.warning(f"Database init warning: {e}")

    # Sidebar upload
    st.sidebar.header('Upload')
    uploaded = st.sidebar.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])

    st.sidebar.header("Data Input Method")
    options = ["Upload File"]
    manual_entry_disabled = uploaded is not None
    if not manual_entry_disabled:
        options.append("Manual Entry")
    else:
        st.sidebar.info("Close the uploaded file to continue in Manual Entry Mode.")

    source_choice = st.sidebar.radio("Select Input Method", options)

    # -------- Session State --------
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "header_row" not in st.session_state:
        st.session_state.header_row = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1

    # -------- Data Ingestion --------
    if source_choice == "Upload File":
        if uploaded is None:
            st.session_state.raw_df = None
            st.session_state.df = None
        else:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded, header=None)
                else:
                    df = pd.read_excel(uploaded, header=None)
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                df = None

            if df is not None and not df.empty:
                # Convert date-like columns to date-only for display
                for col in df.columns:
                    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            temp = pd.to_datetime(df[col], errors='coerce')
                            if temp.notna().all():
                                df[col] = temp.dt.date
                        except Exception:
                            pass

                st.session_state.raw_df = df
                st.session_state.header_row = 0
                st.session_state.df = apply_row_as_header(df, 0)

                # Safe caching for Parquet
                try:
                    df_cache = st.session_state.df.copy()
                    for col in df_cache.columns:
                        if df_cache[col].dtype == 'object' and df_cache[col].apply(lambda x: isinstance(x, pd._libs.tslibs.nattype.NaTType) or isinstance(x, pd.Timestamp) or isinstance(x, pd.Timestamp.date) or isinstance(x, type(pd.Timestamp.now().date()))).any():
                            df_cache[col] = pd.to_datetime(df_cache[col], errors='coerce')
                    save_processed(df_cache, "uploaded_data.parquet")
                except Exception as e:
                    st.info(f"Could not cache uploaded data: {e}")
            else:
                st.warning("Uploaded file is empty or invalid.")
                st.session_state.raw_df = None
                st.session_state.df = None

    elif source_choice == "Manual Entry":
        df = manual_log_entry()
        if df is not None and not df.empty:
            st.session_state.raw_df = df
            st.session_state.df = df  # use as-is, skip header selector
            try:
                save_processed(df, "manual_data.parquet")
            except Exception as e:
                st.info(f"Could not cache manual data: {e}")
        else:
            st.session_state.raw_df = None
            st.session_state.df = None

    # -------- UI: Raw Data Preview + Header Selector --------
    if st.session_state.raw_df is not None and not st.session_state.raw_df.empty:
        st.subheader("Data Preview")

        df = st.session_state.df

        # Header selector only for upload mode
        if source_choice == "Upload File":
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

        # Show updated DataFrame
        df_display = df.reset_index(drop=True).rename_axis("No").rename(lambda x: x + 1, axis=0)
        st.dataframe(df_display.head(50))

        # -------- Preprocess & Embed --------
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
