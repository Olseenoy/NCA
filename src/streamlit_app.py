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

    # --- Handle input method switch ---
    if st.session_state.df is not None and st.session_state.active_input_method != source_choice:
        st.sidebar.warning("Switching input method will terminate ongoing analysis.")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Cancel"):
                st.experimental_rerun()  # refresh page, keep current input
        with col2:
            if st.button("Continue"):
                # Clear all data & reset manual logs
                st.session_state.df = None
                st.session_state.logs = []
                st.session_state.current_log = 1
                st.session_state.active_input_method = source_choice
                st.experimental_rerun()  # refresh page to start new method

    # Ensure the active input method matches sidebar selection
    st.session_state.active_input_method = source_choice

    # --- File Upload ---
    if source_choice == "Upload File":
        if uploaded is None:
            return  # wait until file is uploaded
        df = ingest_file(uploaded)
        if df is not None and not df.empty:
            st.session_state.df = df
            save_processed(df, "uploaded_data.parquet")
        else:
            st.warning("Uploaded file is empty or invalid.")
            st.experimental_rerun()  # reset page if file closed or invalid

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
