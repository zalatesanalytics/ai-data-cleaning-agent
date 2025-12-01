import io
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------
# Helper functions
# ---------------------------------------------


def load_file(uploaded_file) -> pd.DataFrame:
    """Load a file into a pandas DataFrame based on extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    # Fallback: try CSV
    return pd.read_csv(uploaded_file)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to a consistent format and deduplicate."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Handle duplicates by appending a counter suffix
    new_cols = []
    seen: Dict[str, int] = {}
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple cleaning: strip strings, remove fully empty rows/cols."""
    df = df.copy()

    # Strip whitespace in object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows and columns that are completely empty
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Reset index and ensure unique index
    df = df.reset_index(drop=True)

    # Ensure columns are unique again after cleaning
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def concat_harmonized(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Safely concatenate a list of harmonized DataFrames row-wise."""
    cleaned = []
    debug_info = []

    for i, df in enumerate(dfs):
        df2 = normalize_column_names(df)
        df2 = basic_cleaning(df2)

        # Collect debug info about duplicates, if any
        dup_cols = df2.columns[df2.columns.duplicated()]
        if len(dup_cols) > 0:
            debug_info.append(
                f"DataFrame {i} had duplicate columns after cleaning: {list(dup_cols)}"
            )

        if not df2.index.is_unique:
            debug_info.append(
                f"DataFrame {i} had non-unique index; index has been reset."
            )

        cleaned.append(df2)

    if debug_info:
        st.warning("Some issues were detected and automatically fixed during concatenation:")
        for msg in debug_info:
            st.text(f"- {msg}")

    # Final safety: all indexes unique & columns unique
    final_cleaned = []
    for df in cleaned:
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]
        final_cleaned.append(df)

    combined = pd.concat(final_cleaned, ignore_index=True)
    return combined


def download_link_from_df(df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download cleaned data as {filename}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# ---------------------------------------------
# Streamlit app
# ---------------------------------------------

st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

st.title("🧹 AI Data Cleaning & Harmonization Agent")
st.write(
    "Upload one or more tabular datasets (CSV, Excel, JSON, TSV). "
    "The app will standardize column names, perform basic cleaning, "
    "and safely concatenate them into a single combined dataset without "
    "pandas `InvalidIndexError` issues."
)

uploaded_files = st.file_uploader(
    "Upload one or more files",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls", "json", "tsv", "txt"],
)

if not uploaded_files:
    st.info("👆 Upload at least one file to begin.")
    raise SystemExit

# Load each file as DataFrame
raw_dfs: List[pd.DataFrame] = []
st.subheader("1️⃣ Preview of uploaded files")

for i, uf in enumerate(uploaded_files):
    with st.expander(f"File {i+1}: {uf.name}", expanded=i == 0):
        try:
            df = load_file(uf)
        except Exception as e:
            st.error(f"Could not read file `{uf.name}`: {e}")
            continue

        raw_dfs.append(df)

        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(10))

if not raw_dfs:
    st.error("No valid files could be loaded.")
    raise SystemExit

st.subheader("2️⃣ Clean and harmonize")

if st.button("Run cleaning & harmonization"):
    try:
        combined_df = concat_harmonized(raw_dfs)

        st.success(f"Successfully combined {len(raw_dfs)} datasets into a single DataFrame!")
        st.write("Combined shape:", combined_df.shape)

        st.subheader("3️⃣ Combined cleaned dataset (first 100 rows)")
        st.dataframe(combined_df.head(100))

        st.subheader("4️⃣ Download cleaned data")
        download_link_from_df(combined_df, "combined_cleaned_data.csv")

        st.subheader("5️⃣ Quick summary")
        st.write("Column types:")
        st.write(combined_df.dtypes)

        st.write("Numeric summary:")
        st.write(combined_df.describe(include=[np.number]).T)

        st.write("Categorical summary (top categories):")
        cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) == 0:
            st.write("No categorical columns found.")
        else:
            for col in cat_cols:
                st.write(f"**{col}**")
                st.write(combined_df[col].value_counts().head(10))

    except Exception as e:
        st.error(f"An unexpected error occurred during concatenation: {e}")
