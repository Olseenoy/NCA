# src/streamlit_app.py
import os
import sys
import streamlit as st
import pandas as pd

# Ensure src is on path when running from project root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

from ingestion import ingest_file, manual_log_entry, save_processed
from preprocessing import preprocess_df
from embeddings import embed_texts
from clustering import fit_kmeans
from visualization import pareto_plot, cluster_scatter
from pareto import pareto_table
from db import init_db, SessionLocal, CAPA
from rca_engine import rule_based_rca_suggestions, ai_rca_with_fallback
from fishbone_visualizer import visualize_fishbone


def main():
    st.set_page_config(page_title='Smart NC Analyzer', layout='wide')
    st.title('Smart Non-Conformance Analyzer')

    # Initialize database
    init_db()

    # Sidebar upload
    st.sidebar.header('Upload')
    uploaded = st.sidebar.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])

    # Sidebar data input
    st.sidebar.header("Data Input Method")
    source_choice = st.sidebar.radio("Select Input Method", ["Upload File", "Manual Entry"])

    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "active_input_method" not in st.session_state:
        st.session_state.active_input_method = source_choice
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "current_log" not in st.session_state:
        st.session_state.current_log = 1

    # --- Check for input method switch ---
    if st.session_state.df is not None and st.session_state.active_input_method != source_choice:
        st.sidebar.warning("Switching input method will terminate ongoing analysis.")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Cancel"):
                st.experimental_rerun()  # just refresh page
        with col2:
            if st.button("Continue"):
                st.experimental_rerun()  # refresh page to start fresh

    # --- File Upload ---
    if source_choice == "Upload File":
        if uploaded is None:
            return  # do nothing until a file is uploaded
        df = ingest_file(uploaded)
        if df is not None and not df.empty:
            st.session_state.df = df
            save_processed(df, "uploaded_data.parquet")
        else:
            st.warning("Uploaded file is empty or invalid.")
            st.experimental_rerun()  # refresh if file is closed or invalid

    # --- Manual Entry ---
    elif source_choice == "Manual Entry":
        df = manual_log_entry()
        if df is not None and not df.empty:
            st.session_state.df = df
            save_processed(df, "manual_data.parquet")

    # --- Display Raw Data ---
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.subheader("Raw Data Preview")
        df_display = st.session_state.df.reset_index(drop=True).rename_axis("No").rename(lambda x: x + 1, axis=0)
        st.dataframe(df_display.head(50))

        # --- Preprocessing: safe default text columns ---
        default_text_cols = [c for c in st.session_state.df.columns if st.session_state.df[c].dtype == 'object'][:2]
        text_cols = st.multiselect('Text columns to use', options=st.session_state.df.columns.tolist(), default=default_text_cols)




        # --- Preprocess & Embed ---
        default_text_cols = [c for c in df.columns if df[c].dtype == 'object'][:2]
        text_cols = st.multiselect('Text columns to use for embedding', options=df.columns.tolist(), default=default_text_cols)

        if st.button('Preprocess & Embed'):
            p = preprocess_df(df, text_cols)
            st.session_state['processed'] = p
            st.success('Preprocessing complete')
            embeddings = embed_texts(p['clean_text'].tolist())
            st.session_state['embeddings'] = embeddings
            st.success('Embeddings computed')

        # --- Only show clustering, Pareto, SPC after preprocessing ---
        if 'processed' in st.session_state and 'embeddings' in st.session_state:
            p = st.session_state['processed']
            embeddings = st.session_state['embeddings']

            # --- Clustering ---
            st.subheader("Clustering & Visualization")
            if st.button('Cluster & Visualize'):
                km, labels, score, interpretation = fit_kmeans(embeddings)
                st.write(f"Silhouette score: {score:.3f}")
                st.info(interpretation)
                st.session_state['labels'] = labels
                fig = cluster_scatter(embeddings, labels)
                st.plotly_chart(fig, use_container_width=True)

            # --- Pareto Analysis ---
            st.subheader("Pareto Analysis")
            cat_col = st.selectbox('Select column for Pareto', options=p.columns.tolist())
            if st.button('Show Pareto'):
                tab = pareto_table(p, cat_col)
                fig = pareto_plot(tab)
                st.plotly_chart(fig, use_container_width=True)

            # --- SPC Section ---
            st.subheader("Statistical Process Control (SPC)")
            num_cols = p.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                spc_col = st.selectbox('Select numeric column for SPC', options=num_cols)
                if st.button('Show SPC Chart'):
                    fig_spc = plot_spc_chart(p, spc_col)
                    st.plotly_chart(fig_spc, use_container_width=True)
            else:
                st.info("No numeric columns available for SPC analysis.")

            # --- Root Cause Analysis (RCA) ---
            st.subheader("Root Cause Analysis (RCA)")
            idx = st.number_input('Pick row index to analyze', min_value=0, max_value=len(p)-1, value=0)
            row = p.iloc[int(idx)]
            st.markdown("**Selected row preview:**")
            st.write(row.get('combined_text', ''))

            mode = st.radio("RCA Mode", options=["AI-Powered (LLM)", "Rule-Based (fallback)"])

            if st.button("Run RCA"):
                with st.spinner("Running RCA..."):
                    if mode == "AI-Powered (LLM)":
                        result = ai_rca_with_fallback(row['combined_text'], row['clean_text'])
                    else:
                        fb = rule_based_rca_suggestions(row['clean_text'])
                        result = {"from": "rule_based", "fishbone": fb}

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

            # --- Manual 5-Whys & CAPA creation ---
            st.markdown("---")
            st.subheader("Manual 5-Whys & CAPA creation")
            manual_whys = []
            for i in range(5):
                manual_whys.append(st.text_input(f"Why {i+1}", key=f"manual_why_{i}"))

            with st.form("capa_form"):
                issue_id = st.text_input("Issue ID", value=f"issue-{idx}")
                desc = st.text_area("Description", value=row.get("combined_text", ""))
                corrective = st.text_area("Corrective Action")
                preventive = st.text_area("Preventive Action")
                owner = st.text_input("Owner")
                due_days = st.number_input("Due in (days)", min_value=1, max_value=365, value=14)
                submitted = st.form_submit_button("Create CAPA")
                if submitted:
                    db = SessionLocal()
                    from datetime import datetime, timedelta
                    due_date = datetime.utcnow() + timedelta(days=int(due_days))
                    capa = CAPA(issue_id=issue_id,
                                description=desc,
                                corrective_action=(corrective + "\n\nPreventive:\n" + preventive),
                                owner=owner,
                                due_date=due_date)
                    db.add(capa)
                    db.commit()
                    st.success("CAPA created and saved to DB")


if __name__ == "__main__":
    main()
