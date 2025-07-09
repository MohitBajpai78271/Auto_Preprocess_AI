import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from explain import explain_preprocessing

st.set_page_config(page_title="CSV Preprocessor", layout="wide")
st.title("📊 AutoPreprocessAI – CSV Preprocessor Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Sidebar configuration
st.sidebar.header("⚙️ Preprocessing Settings")

# Only show settings if file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    all_columns = df.columns.tolist()

    # Sidebar inputs
    drop_cols = st.sidebar.multiselect("🗑️ Columns to Drop", options=all_columns)
    num_strategy = st.sidebar.selectbox("🔢 Numerical Imputation Strategy", ["median", "mean", "most_frequent"])
    cat_strategy = st.sidebar.selectbox("🔤 Categorical Imputation Strategy", ["most_frequent", "constant", "drop"])
    apply_scaling = st.sidebar.checkbox("📏 Apply Standard Scaling", value=True)
    apply_encoding = st.sidebar.checkbox("🎭 One-hot Encode Categorical Variables", value=True)

    # Run preprocessing
    if st.button("🚀 Run Auto Preprocessing"):
        cleaned_df, summary = preprocess_data(
            df,
            drop_cols=drop_cols,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            apply_scaling=apply_scaling,
            apply_encoding=apply_encoding
        )

        st.session_state['cleaned_df'] = cleaned_df
        st.session_state['summary'] = summary

        st.subheader("✅ Cleaned Dataset")
        st.dataframe(cleaned_df.head())

        st.subheader("📝 Preprocessing Summary")
        for step in summary:
            st.markdown(f"- {step}")

    # AI Explanation button (only after preprocessing)
    if 'summary' in st.session_state and st.session_state.get('summary'):
        if st.button("🤖 Explain with AI"):
            explanation = explain_preprocessing(
                st.session_state['summary'],
                dataset_name=uploaded_file.name
            )
            st.subheader("🧠 AI Explanation of Preprocessing")
            st.markdown(explanation)

            # Download cleaned CSV
        csv = st.session_state['cleaned_df'].to_csv(index=False).encode("utf-8")
        st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name=f"cleaned_{uploaded_file.name}",
                mime="text/csv"
         )

         # Download preprocessing summary
        summary_text = "\n".join(f"- {step}" for step in st.session_state['summary'])
        st.download_button(
                label="📝 Download Preprocessing Summary",
                data=summary_text,
                file_name="preprocessing_summary.txt",
                mime="text/plain"
        )

    elif 'cleaned_df' not in st.session_state:
        st.info("ℹ️ Run preprocessing to enable AI explanation.")

else:
    st.info("👈 Please upload a CSV file to get started.")
