import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Amazon Review Analysis Agent",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Amazon Review Analysis Agent")
st.write(
    "Upload an Amazon review CSV file. The agent will first observe the dataset "
    "by checking fields, missing values, and basic structure."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2. Basic Observation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        missing_values = int(df.isna().sum().sum())
        st.metric("Missing Values", missing_values)

    st.subheader("3. Column List")
    st.write(list(df.columns))

    st.subheader("4. Missing Values by Column")
    missing_by_column = df.isna().sum().reset_index()
    missing_by_column.columns = ["column", "missing_count"]
    st.dataframe(missing_by_column, use_container_width=True)

else:
    st.info("Please upload a CSV file to start.")