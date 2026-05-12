import streamlit as st
import pandas as pd

from agent import run_review_analysis_agent

st.set_page_config(
    page_title="Amazon Review Analysis Agent",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Amazon Review Analysis Agent")
st.write(
    "Upload an Amazon review CSV file. The agent will observe the dataset, "
    "decide an analysis path, classify review issues, and recommend QC actions."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run Agent Analysis"):
        result = run_review_analysis_agent(df)

        st.subheader("2. Agent Observation")
        observation = result["observation"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", observation["total_rows"])

        with col2:
            st.metric("Columns", observation["total_columns"])

        with col3:
            st.metric("Missing Comments", f"{observation['comment_missing_rate']}%")

        with col4:
            st.metric("Low Rating Reviews", f"{observation['low_rating_rate']}%")

        st.write("Detected columns:")
        st.write(observation["columns"])

        st.subheader("3. Agent Decision")
        decision = result["decision"]

        st.write(f"Analysis focus: **{decision['analysis_focus']}**")

        if decision["warnings"]:
            for warning in decision["warnings"]:
                st.warning(warning)
        else:
            st.success("No major data quality warnings detected.")

        if result["classified_reviews"] is None:
            st.error("The uploaded CSV is missing required columns. Please check the dataset format.")

        else:
            classified_reviews = result["classified_reviews"]
            category_statistics = result["category_statistics"]
            qc_recommendations = result["qc_recommendations"]

            st.subheader("4. Review Classification Results")
            st.dataframe(
                classified_reviews[
                    [
                        "review_id",
                        "rating",
                        "review_title",
                        "agent_category",
                        "agent_confidence",
                        "agent_evidence",
                    ]
                ],
                use_container_width=True
            )

            st.subheader("5. Category Statistics")
            st.dataframe(category_statistics, use_container_width=True)

            chart_df = category_statistics.set_index("category")["count"]
            st.bar_chart(chart_df)

            st.subheader("6. QC Recommendations")

            if qc_recommendations:
                for item in qc_recommendations:
                    st.markdown(
                        f"""
                        **Category:** `{item['category']}`  
                        **Count:** {item['count']} reviews  
                        **Share:** {item['percentage']}%  
                        **Recommended action:** {item['recommendation']}
                        """
                    )
                    st.divider()
            else:
                st.info("No defect-specific QC recommendations were generated.")

else:
    st.info("Please upload a CSV file to start.")