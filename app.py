import streamlit as st
import pandas as pd
import numpy as np
import os

from utils.schema_matcher import suggest_column_groups, build_canonical_mapping
from utils.cleaning import basic_cleaning_report, apply_basic_cleaning
from utils.eda import generate_eda_summary

st.set_page_config(
    page_title="AI Data-Cleaning & Integration Agent",
    layout="wide"
)

st.title("🧹 AI Data-Cleaning & Integration Agent")

st.markdown(
    """
This agent helps you:
- Upload **multiple datasets** (CSV/Excel for now)
- Detect similar column names (e.g., `sex` vs `gender`, `Age` vs `age`)
- Harmonize schemas
- Apply basic data cleaning
- Run simple EDA

More formats (STATA, SPSS, etc.) can be added later.
"""
)

# ---------- Helper functions ----------

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        st.warning(f"Unsupported file type for {uploaded_file.name}. Only CSV/Excel in this demo.")
        return None

# ---------- Data upload section ----------

st.sidebar.header("1. Upload datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

use_sample = st.sidebar.checkbox("Use sample synthetic data (demo)", value=False)

dataframes = []
dataset_names = []

if use_sample:
    st.sidebar.success("Using sample synthetic datasets from /data")
    for fname in ["sample_dataset1.csv", "sample_dataset2.csv", "sample_dataset3.csv"]:
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            dataframes.append(df)
            dataset_names.append(fname)
        else:
            st.sidebar.warning(f"Sample file not found: {fpath}")
elif uploaded_files:
    for f in uploaded_files:
        df = load_file(f)
        if df is not None:
            dataframes.append(df)
            dataset_names.append(f.name)

if not dataframes:
    st.info("Upload at least one dataset from the sidebar or enable sample synthetic data.")
    st.stop()

st.subheader("Preview of uploaded datasets")
for name, df in zip(dataset_names, dataframes):
    st.markdown(f"#### 📄 {name}")
    st.write(df.head())

# ---------- Schema matching ----------

st.header("2. Schema matching (column-name intelligence)")

all_columns = [df.columns.tolist() for df in dataframes]
column_groups = suggest_column_groups(all_columns, threshold=85)
mapping = build_canonical_mapping(column_groups)

st.markdown("**Suggested column groups (similar names):**")
for group in column_groups:
    if len(group) > 1:
        st.write(" • " + ", ".join(group))

st.markdown(
    """
The app will **standardize column names within each group** (using the first name in each group as the canonical label).
You can review this suggestion above. A more advanced version could allow manual editing of groupings.
"""
)

# Apply canonical mapping to every dataframe
harmonized_dfs = []
for df in dataframes:
    df_renamed = df.rename(columns=mapping)
    harmonized_dfs.append(df_renamed)

# ---------- Combine datasets (append) ----------

st.header("3. Combine datasets")

combine_mode = st.radio(
    "How would you like to combine the datasets?",
    ["Append rows (stack)", "Just keep separate (no combine yet)"]
)

combined_df = None
if combine_mode == "Append rows (stack)":
    combined_df = pd.concat(harmonized_dfs, ignore_index=True)
    st.success(f"Combined dataset shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
    st.dataframe(combined_df.head())
else:
    st.info("Datasets will be cleaned individually without combining.")
    # For EDA/cleaning we will use only the first dataset if not combined
    combined_df = harmonized_dfs[0]

# ---------- Data cleaning ----------

st.header("4. Data quality check & cleaning")

if st.checkbox("Show data quality report"):
    report = basic_cleaning_report(combined_df)
    st.subheader("Missing values (per column)")
    st.json(report["missing_counts"])

    st.subheader("Outliers detected (per numeric column)")
    st.json(report["outliers"])

    st.subheader("Logical issues (age, income, etc.)")
    st.json(report["logical_issues"])

st.markdown("Click the button below to apply **basic cleaning rules** (convert numeric-like fields, fix age and negative incomes).")

if st.button("Apply basic cleaning"):
    cleaned_df = apply_basic_cleaning(combined_df)
    st.success("Basic cleaning applied.")
    st.dataframe(cleaned_df.head())

    # Store cleaned in session state for EDA and download
    st.session_state["cleaned_df"] = cleaned_df
else:
    st.session_state["cleaned_df"] = combined_df

# ---------- EDA ----------

st.header("5. Exploratory Data Analysis (EDA)")

df_for_eda = st.session_state.get("cleaned_df", combined_df)
eda_summary = generate_eda_summary(df_for_eda)

st.subheader("Shape")
st.write(eda_summary["shape"])

st.subheader("Column types")
st.json(eda_summary["columns"])

st.subheader("Descriptive statistics")
st.write(pd.DataFrame.from_dict(eda_summary["describe"], orient="index"))

# Simple plots
st.subheader("Quick plots")

numeric_cols = df_for_eda.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    chosen_num = st.selectbox("Choose a numeric column for histogram", numeric_cols)
    st.bar_chart(df_for_eda[chosen_num].dropna())
else:
    st.info("No numeric columns available for plotting.")

# ---------- Download cleaned dataset ----------

st.header("6. Download cleaned / combined dataset")

if df_for_eda is not None:
    csv_data = df_for_eda.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned dataset as CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
