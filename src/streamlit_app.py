# src/streamlit_app.py
import os
import sys
# Ensure the local `src/` directory is on sys.path so modules like `ingestion`,
# `preprocessing`, `config`, etc. can be imported when running
# `streamlit run src/streamlit_app.py` from the project root.
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

import streamlit as st
import pandas as pd
from ingestion import ingest_csv, save_processed
from preprocessing import preprocess_df
from embeddings import embed_texts, build_faiss_index, query_top_k
from clustering import fit_kmeans
from visualization import pareto_plot, cluster_scatter
from pareto import pareto_table
from rca_engine import rule_based_rca_suggestions, five_whys
from db import init_db, SessionLocal, CAPA
# LLM RCA
from llm_rca import generate_rca_with_llm


def main():
    st.set_page_config(page_title='Smart NC Analyzer', layout='wide')
    st.title('Smart Non-Conformance Analyzer')

    init_db()

    st.sidebar.header('Upload')
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f'Loaded {len(df)} rows')
        st.session_state['raw_df'] = df

    if 'raw_df' in st.session_state:
        df = st.session_state['raw_df']
        st.dataframe(df.head())

        # Preprocess
        text_cols = st.multiselect('Text columns to use', options=df.columns.tolist(), default=[c for c in df.columns if df[c].dtype == 'object'][:2])
        if st.button('Preprocess & Embed'):
            p = preprocess_df(df, text_cols)
            st.session_state['processed'] = p
            st.success('Preprocessed')
            embeddings = embed_texts(p['clean_text'].tolist())
            st.session_state['embeddings'] = embeddings
            st.success('Embeddings computed')

    if 'processed' in st.session_state and 'embeddings' in st.session_state:
        p = st.session_state['processed']
        embeddings = st.session_state['embeddings']
        # Clustering
        if st.button('Cluster & Visualize'):
            km, labels, score = fit_kmeans(embeddings)
            st.write(f'Silhouette score: {score:.3f}')
            st.session_state['labels'] = labels
            fig = cluster_scatter(embeddings, labels)
            st.plotly_chart(fig, use_container_width=True)

        # Pareto
        cat_col = st.selectbox('Select column for Pareto', options=p.columns.tolist())
        if st.button('Show Pareto'):
            tab = pareto_table(p, cat_col)
            fig = pareto_plot(tab)
            st.plotly_chart(fig, use_container_width=True)

        # RCA
        idx = st.number_input('Pick row index to analyze', min_value=0, max_value=len(p)-1, value=0)
        if st.button('Run RCA on row'):
            row = p.iloc[int(idx)]
            suggestions = rule_based_rca_suggestions(row['clean_text'])
            st.json(suggestions)
            st.write('Run 5Whys?')
            whys = []
            for i in range(5):
                ans = st.text_input(f'Why {i+1}?', key=f'why_{i}')
                whys.append(ans)
            if st.button('Save 5Whys'):
                chain = five_whys(row['combined_text'], whys)
                st.write(chain)

            # Automated LLM RCA
            st.write('Automated RCA (LLM)')
            if st.button('Generate automated RCA & CAPA (LLM)'):
                with st.spinner('Contacting LLM...'):
                    try:
                        rca_result = generate_rca_with_llm(row['combined_text'])
                        st.subheader('LLM suggested root causes')
                        st.json(rca_result.get('root_causes'))
                        st.subheader('LLM suggested 5-Whys chain')
                        st.write(rca_result.get('five_whys'))
                        st.subheader('LLM suggested CAPA')
                        st.write(rca_result.get('capa'))
                    except Exception as e:
                        st.error(f'LLM RCA failed: {e}')

        # CAPA quick create
        st.header('CAPA')
        with st.form('capa_form'):
            issue_id = st.text_input('Issue ID')
            desc = st.text_area('Description')
            ca = st.text_area('Corrective Action')
            owner = st.text_input('Owner')
            submit = st.form_submit_button('Create CAPA')
            if submit:
                db = SessionLocal()
                capa = CAPA(issue_id=issue_id, description=desc, corrective_action=ca, owner=owner)
                db.add(capa)
                db.commit()
                st.success('CAPA created')

if __name__ == '__main__':
    main()
