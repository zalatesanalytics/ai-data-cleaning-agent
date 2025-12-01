import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports for advanced formats / ML
try:
    import pyreadstat  # for .sav
except ImportError:
    pyreadstat = None

# =========================================
# 1. Synthetic demo dataset
# =========================================

def generate_synthetic_dataset(n: int = 300) -> pd.DataFrame:
    """Generate a synthetic dataset with missing values, extremes, and inconsistencies."""
    rng = np.random.default_rng(42)
    ages = rng.integers(-5, 150, size=n)  # includes impossible ages
    income = rng.normal(3000, 1500, size=n)
    income[rng.choice(n, size=15, replace=False)] = -100  # negative income
    income[rng.choice(n, size=5, replace=False)] = 2_000_000  # extreme income

    gender = rng.choice(["Male", "Female", "Other", None, "Unknown"], size=n)
    employed = rng.choice(["Yes", "No"], size=n)
    edu = rng.choice(["None", "Primary", "Secondary", "College", "University"], size=n)

    # Introduce contradictions
    for i in range(0, n, 25):
        ages[i] = 8
        edu[i] = "University"

    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "Age": ages,
            "gender": gender,
            "income_monthly": income,
            "employed": employed,
            "education_level": edu,
        }
    )

    # Add missingness
    for col in ["Age", "income_monthly"]:
        df.loc[rng.choice(n, size=20, replace=False), col] = np.nan

    # Placeholder values
    df.loc[rng.choice(n, size=10, replace=False), "income_monthly"] = 999999
    df.loc[rng.choice(n, size=10, replace=False), "gender"] = "?"

    return df


# =========================================
# 2. File loading helpers
# =========================================

ALLOWED_EXTS = [".csv", ".xlsx", ".xls", ".dta", ".sav", ".json", ".pkl", ".pickle", ".tsv", ".txt"]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a Streamlit-uploaded file into a pandas DataFrame based on extension."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".dta"):
        return pd.read_stata(uploaded_file)
    if name.endswith(".sav"):
        if pyreadstat is None:
            raise ImportError("pyreadstat is required to read .sav files. Please install it.")
        df, _ = pyreadstat.read_sav(uploaded_file)
        return df
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".pkl") or name.endswith(".pickle"):
        return pickle.load(uploaded_file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")

    # Fallback: try CSV
    return pd.read_csv(uploaded_file)


def load_local_sample(path: str) -> pd.DataFrame:
    """Load a sample file from ./data directory by path."""
    name = path.lower()

    if name.endswith(".csv"):
        return pd.read_csv(path)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path)
    if name.endswith(".dta"):
        return pd.read_stata(path)
    if name.endswith(".sav"):
        if pyreadstat is None:
            raise ImportError("pyreadstat is required to read .sav files. Please install it.")
        df, _ = pyreadstat.read_sav(path)
        return df
    if name.endswith(".json"):
        return pd.read_json(path)
    if name.endswith(".pkl") or name.endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def discover_sample_datasets(data_dir: str = "data") -> Dict[str, str]:
    """Return dict {display_name: path} for sample files in ./data."""
    samples: Dict[str, str] = {}
    if not os.path.isdir(data_dir):
        return samples

    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        if any(fname.lower().endswith(ext) for ext in ALLOWED_EXTS):
            samples[fname] = path
    return samples


# =========================================
# 3. Schema & name normalization
# =========================================

def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Standardize column names to a consistent format, but keep original in a mapping.
    Returns (normalized_df, original_to_normalized_map).
    """
    df = df.copy()
    original_cols = list(df.columns)
    norm_cols = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # handle duplicates by adding suffix
    new_cols: List[str] = []
    seen: Dict[str, int] = {}
    for col in norm_cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")

    df.columns = new_cols
    col_map = dict(zip(original_cols, new_cols))
    return df, col_map


def get_schema_summary(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a tidy schema summary: file, variable, dtype."""
    rows = []
    for fname, df in dfs.items():
        for col in df.columns:
            rows.append({"file": fname, "variable": col, "dtype": str(df[col].dtype)})
    return pd.DataFrame(rows)


def suggest_similar_columns(dfs: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str]]:
    """
    Suggest potentially similar columns across files based on simple heuristics:
    - lower-case comparison without underscores
    - synonym mapping for common variable names (age, sex/gender, id)
    Returns tuples of (file_name, col_in_that_file, matching_col_from_some_other_file).
    """
    import difflib

    # Collect all columns with file names
    col_entries = []  # (file, col)
    for fname, df in dfs.items():
        for col in df.columns:
            col_entries.append((fname, col))

    suggestions: List[Tuple[str, str, str]] = []
    synonyms = {
        "sex": ["gender", "sex", "sexe"],
        "gender": ["gender", "sex", "sexe"],
        "age": ["age", "age_years", "years", "age_yrs"],
        "id": ["id", "hhid", "household_id", "respondent_id"],
        "income": ["income", "income_monthly", "income1", "salary", "wage"],
    }

    def base_name(c: str) -> str:
        return c.lower().replace("_", "").replace(" ", "")

    for i in range(len(col_entries)):
        f1, c1 = col_entries[i]
        for j in range(i + 1, len(col_entries)):
            f2, c2 = col_entries[j]
            if f1 == f2:
                continue

            b1, b2 = base_name(c1), base_name(c2)
            ratio = difflib.SequenceMatcher(None, b1, b2).ratio()

            # synonym-based
            syn_match = False
            for _, variants in synonyms.items():
                if c1.lower() in variants and c2.lower() in variants:
                    syn_match = True
                    break

            if syn_match or ratio > 0.8:
                suggestions.append((f1, c1, c2))

    # Deduplicate suggestions
    unique_suggestions = list(dict.fromkeys(suggestions))
    return unique_suggestions


# =========================================
# 4. Data cleaning utilities
# =========================================

PLACEHOLDER_VALUES = {"?", "NA", "N/A", "999", "9999", "Unknown", "unknown"}


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic structural cleaning: strip strings, remove fully empty rows/cols, reset index."""
    df = df.copy()

    # Strip whitespace in object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(list(PLACEHOLDER_VALUES), np.nan)

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def detect_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness summary per column."""
    total = len(df)
    miss = df.isna().sum()
    return pd.DataFrame(
        {
            "missing_count": miss,
            "missing_pct": (miss / total * 100).round(2),
            "dtype": df.dtypes.astype(str),
        }
    ).sort_values("missing_pct", ascending=False)


def detect_numeric_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential extreme values for numeric columns."""
    rows = []
    num_df = df.select_dtypes(include=[np.number])
    for col in num_df.columns:
        series = num_df[col].dropna()
        if series.empty:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()

        rule_note = ""
        if "age" in col.lower():
            rule_note = "Age rule (0–120)"
            impossible = ((series < 0) | (series > 120)).sum()
        elif "income" in col.lower() or "salary" in col.lower():
            rule_note = "Income rule (>=0 & < 1,000,000)"
            impossible = ((series < 0) | (series > 1_000_000)).sum()
        else:
            impossible = np.nan

        rows.append(
            {
                "variable": col,
                "n": len(series),
                "outliers_iqr": int(outliers),
                "rule_note": rule_note,
                "impossible_rule_violations": (
                    int(impossible) if not np.isnan(impossible) else np.nan
                ),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def detect_duplicates(df: pd.DataFrame, id_cols: List[str]) -> Dict[str, int]:
    """Return count of duplicates for each ID column."""
    result = {}
    for col in id_cols:
        if col in df.columns:
            dup_count = df.duplicated(subset=[col]).sum()
            result[col] = int(dup_count)
    return result


def detect_logical_inconsistencies(df: pd.DataFrame) -> List[str]:
    """Simple logical rules for inconsistencies."""
    messages: List[str] = []
    lower_name_map = {c.lower(): c for c in df.columns}

    # Age & education rule
    if "age" in lower_name_map and "education_level" in lower_name_map:
        age_col = lower_name_map["age"]
        edu_col = lower_name_map["education_level"]
        subset = df[[age_col, edu_col]].copy()
        with np.errstate(invalid="ignore"):
            mask = (subset[age_col] < 15) & subset[edu_col].isin(["College", "University"])
        count = int(mask.sum())
        if count > 0:
            messages.append(
                f"{count} records: age < 15 but education level is College/University."
            )

    # Employed vs income
    emp_col = None
    income_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"employed", "employment_status"}:
            emp_col = c
        if "income" in lc or "salary" in lc:
            income_col = c

    if emp_col and income_col:
        subset = df[[emp_col, income_col]].copy()
        with np.errstate(invalid="ignore"):
            mask = (subset[emp_col].astype(str).str.lower() == "yes") & (
                (subset[income_col] <= 0) | subset[income_col].isna()
            )
        count = int(mask.sum())
        if count > 0:
            messages.append(
                f"{count} records: employed='Yes' but income is 0 or missing."
            )

    return messages


# =========================================
# 5. Anomaly detection (optional ML)
# =========================================

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """Run Isolation Forest on numeric columns, return anomaly flag Series."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        st.warning("scikit-learn is not installed; Isolation Forest is unavailable.")
        return pd.Series([0] * len(df), index=df.index)

    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 1 or num_df.shape[0] < 10:
        st.info("Not enough numeric data for Isolation Forest.")
        return pd.Series([0] * len(df), index=df.index)

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(num_df)
    anomaly = pd.Series(0, index=df.index)
    anomaly.loc[num_df.index] = (preds == -1).astype(int)
    return anomaly


# =========================================
# 6. Integration: append vs merge
# =========================================

def harmonize_columns(
    dfs: Dict[str, pd.DataFrame],
    mappings: Dict[str, Dict[str, str]],
) -> Dict[str, pd.DataFrame]:
    """Apply user-defined mappings: mappings[file][old_col] = new_col."""
    out: Dict[str, pd.DataFrame] = {}
    for fname, df in dfs.items():
        df = df.copy()
        if fname in mappings:
            df = df.rename(columns=mappings[fname])
        df = df.loc[:, ~df.columns.duplicated()]
        out[fname] = df
    return out


def safely_append(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Vertically append multiple DataFrames with safety checks."""
    cleaned = []
    for df in dfs.values():
        df = basic_cleaning(df)
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]
        cleaned.append(df)

    combined = pd.concat(cleaned, ignore_index=True, sort=False)
    return combined


def safely_merge(dfs: Dict[str, pd.DataFrame], key: str, how: str = "outer") -> pd.DataFrame:
    """Horizontally merge multiple DataFrames on a key."""
    cleaned = []
    for df in dfs.values():
        df = basic_cleaning(df)
        cleaned.append(df)

    merged: pd.DataFrame | None = None
    for df in cleaned:
        if key not in df.columns:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=key, how=how, suffixes=("", "_dup"))

    if merged is None:
        st.error(f"No dataset contained the key column '{key}'.")
        return pd.DataFrame()

    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reset_index(drop=True)
    return merged


# =========================================
# 7. EDA helpers
# =========================================

def show_basic_eda(df: pd.DataFrame):
    st.markdown("#### 📊 Descriptive Statistics")
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        st.dataframe(df.describe(include=[np.number]).T)
    else:
        st.write("No numeric columns found.")

    st.markdown("#### 📋 Categorical Frequencies (Top 10)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        st.write("No categorical columns detected.")
    else:
        for col in cat_cols:
            with st.expander(f"Variable: {col}", expanded=False):
                st.write(df[col].value_counts(dropna=False).head(10))

    st.markdown("#### 🔗 Correlation Heatmap (numeric)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
    else:
        st.write("Not enough numeric variables for correlation heatmap.")

    st.markdown("#### 📈 Simple Histogram")
    num_cols = list(num_df.columns)
    if num_cols:
        col_to_plot = st.selectbox("Select numeric variable for histogram", num_cols)
        st.bar_chart(num_df[col_to_plot].dropna().value_counts().sort_index())
    else:
        st.write("No numeric columns available for plotting.")


# =========================================
# 8. Streamlit App – Dashboard Layout
# =========================================

st.set_page_config(page_title="AI Data Cleaning & Integration Agent", layout="wide")

st.title("🧹 AI Data-Cleaning, Integration & Analysis Agent")

st.write(
    """
This agent ingests multiple datasets (CSV, Excel, Stata, SPSS, JSON, pickle), 
detects schema similarities, diagnoses data-quality issues, supports append/merge 
integration, and prepares clean, analysis-ready data with EDA and optional 
ML-based anomaly detection.
"""
)

# Sidebar: data source
st.sidebar.header("Data Source")

data_mode = st.sidebar.radio(
    "Choose data source:",
    ["Upload your own files", "Built-in sample files (./data)", "Synthetic demo"],
)

dfs_raw: Dict[str, pd.DataFrame] = {}

# --- Mode 1: Upload files ---
if data_mode == "Upload your own files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more datasets",
        accept_multiple_files=True,
        type=[ext.replace(".", "") for ext in ALLOWED_EXTS],
    )
    if uploaded_files:
        for uf in uploaded_files:
            try:
                df = load_uploaded_file(uf)
                dfs_raw[uf.name] = df
            except Exception as e:
                st.error(f"❌ Could not read file `{uf.name}`: {e}")

# --- Mode 2: Built-in sample files from ./data ---
elif data_mode == "Built-in sample files (./data)":
    samples = discover_sample_datasets("data")
    if not samples:
        st.sidebar.warning("No sample files found in ./data folder.")
    else:
        selected_samples = st.sidebar.multiselect(
            "Select one or more sample files",
            options=list(samples.keys()),
        )
        for fname in selected_samples:
            path = samples[fname]
            try:
                df = load_local_sample(path)
                dfs_raw[fname] = df
            except Exception as e:
                st.error(f"❌ Could not read sample `{fname}`: {e}")

# --- Mode 3: Synthetic demo ---
elif data_mode == "Synthetic demo":
    # Provide 1–3 synthetic datasets to test append/merge logic
    n_datasets = st.sidebar.slider("Number of synthetic datasets", 1, 3, 2)
    for i in range(n_datasets):
        df = generate_synthetic_dataset()
        dfs_raw[f"synthetic_{i+1}.csv"] = df

# If nothing loaded
if not dfs_raw:
    st.info("No datasets loaded yet. Adjust your data source on the left.")
    st.stop()

# ==========================
# Tabs for nicer dashboard
# ==========================
tab_overview, tab_schema, tab_quality, tab_eda, tab_ml = st.tabs(
    ["📋 Overview", "🧩 Schema & Matching", "✅ Data Quality", "📊 EDA", "🤖 ML Anomalies"]
)

# --- OVERVIEW TAB ---
with tab_overview:
    st.subheader("1️⃣ Preview of Loaded Datasets")

    for fname, df in dfs_raw.items():
        with st.expander(f"File: {fname} (shape={df.shape[0]} rows, {df.shape[1]} cols)", expanded=False):
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(10))

    # Normalize columns for internal use
    dfs_norm: Dict[str, pd.DataFrame] = {}
    col_maps: Dict[str, Dict[str, str]] = {}
    for fname, df in dfs_raw.items():
        ndf, cmap = normalize_column_names(df)
        dfs_norm[fname] = ndf
        col_maps[fname] = cmap

    # Basic metrics (based on concatenated view just for counts)
    tmp_concat = safely_append(dfs_norm)
    total_rows = tmp_concat.shape[0]
    total_cols = tmp_concat.shape[1]
    total_missing = int(tmp_concat.isna().sum().sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total rows (all datasets combined)", f"{total_rows:,}")
    m2.metric("Total columns (after normalization)", f"{total_cols:,}")
    m3.metric("Total missing cells", f"{total_missing:,}")

# --- SCHEMA & MATCHING TAB ---
with tab_schema:
    st.subheader("2️⃣ Schema Summary")
    schema_df = get_schema_summary(dfs_norm)
    st.dataframe(schema_df)

    st.markdown("### 3️⃣ Column Similarity & Harmonization")

    harmonization_mappings: Dict[str, Dict[str, str]] = {fname: {} for fname in dfs_norm.keys()}
    suggestions = suggest_similar_columns(dfs_norm)

    if suggestions:
        st.write(
            "The agent detected potentially similar variables across datasets. "
            "Tick the ones that should be treated as the same variable."
        )
        for idx, (f1, c1, c2) in enumerate(suggestions):
            label = f"Treat `{c1}` in **{f1}** and `{c2}` in the other file as the same variable?"
            if st.checkbox(label, key=f"suggest_{idx}"):
                base = c1
                harmonization_mappings[f1][c1] = base
                for fname, df in dfs_norm.items():
                    if c2 in df.columns:
                        harmonization_mappings[fname][c2] = base
    else:
        st.write("No strong column similarity suggestions found. You can still harmonize manually.")

    st.markdown("**Manual harmonization (optional)**")
    with st.expander("Map columns manually to a standard name", expanded=False):
        for fname, df in dfs_norm.items():
            st.markdown(f"**File: {fname}**")
            for col in df.columns:
                new_name = st.text_input(
                    f"Standard name for `{col}` in {fname} (leave blank to keep)",
                    value="",
                    key=f"manual_{fname}_{col}",
                )
                if new_name:
                    harmonization_mappings[fname][col] = new_name

    if st.button("Apply harmonization & prepare for integration"):
        st.session_state["harmonized_dfs"] = harmonize_columns(dfs_norm, harmonization_mappings)
        st.success("Harmonization applied.")
    else:
        if "harmonized_dfs" not in st.session_state:
            st.session_state["harmonized_dfs"] = dfs_norm

harmonized_dfs: Dict[str, pd.DataFrame] = st.session_state["harmonized_dfs"]

# --- DATA QUALITY TAB (includes integration step) ---
with tab_quality:
    st.subheader("4️⃣ Integration Mode")

    integration_mode = st.radio(
        "How should the datasets be integrated?",
        ["Append (stack rows)", "Merge (join on ID)"],
        key="integration_mode_radio",
    )

    integrated_df = pd.DataFrame()

    if integration_mode == "Append (stack rows)":
        if st.button("Run append integration", key="btn_append"):
            integrated_df = safely_append(harmonized_dfs)
            st.success(f"Appended {len(harmonized_dfs)} datasets. Result shape: {integrated_df.shape}")
    else:
        st.write("Select the key column for merging (e.g., `id`, `hhid`, `household_id`).")
        candidate_ids = set()
        for df in harmonized_dfs.values():
            for c in df.columns:
                if any(k in c.lower() for k in ["id", "hhid", "household"]):
                    candidate_ids.add(c)
        candidate_ids = sorted(candidate_ids)
        if not candidate_ids:
            st.warning("No obvious ID columns found. You can still type one manually.")
        key = st.text_input("Merge key column name:", value=candidate_ids[0] if candidate_ids else "")
        merge_type = st.selectbox("Merge type", ["outer", "inner", "left", "right"])

        if st.button("Run merge integration", key="btn_merge"):
            if not key:
                st.error("Please specify a key column for merging.")
            else:
                integrated_df = safely_merge(harmonized_dfs, key=key, how=merge_type)
                if not integrated_df.empty:
                    st.success(f"Merged datasets on `{key}`. Result shape: {integrated_df.shape}")

    if integrated_df.empty:
        st.info("Run integration above to see data-quality diagnostics.")
        st.stop()

    st.markdown("### 5️⃣ Data-Quality Assessment")

    with st.expander("Missing values summary", expanded=True):
        miss_df = detect_missingness(integrated_df)
        st.dataframe(miss_df)

    with st.expander("Numeric extremes & impossible values", expanded=True):
        ext_df = detect_numeric_extremes(integrated_df)
        st.dataframe(ext_df)

    with st.expander("Duplicate IDs", expanded=False):
        candidate_ids = [c for c in integrated_df.columns if "id" in c.lower()]
        if candidate_ids:
            chosen_ids = st.multiselect(
                "Select ID columns to check for duplicates", candidate_ids, default=candidate_ids
            )
            dup_info = detect_duplicates(integrated_df, chosen_ids)
            if dup_info:
                st.write(dup_info)
            else:
                st.write("No duplicates detected for selected ID columns.")
        else:
            st.write("No ID-like columns found for duplicate checks.")

    with st.expander("Logical inconsistencies", expanded=False):
        msgs = detect_logical_inconsistencies(integrated_df)
        if msgs:
            for m in msgs:
                st.warning(m)
        else:
            st.write("No simple logical inconsistencies detected by current rules.")

    st.markdown(
        "Detected potential issues above. You can choose to run an **automatic cleaning pass** "
        "or manually export and fix."
    )

    auto_clean = st.checkbox("Apply automatic cleaning (convert numerics, cap ages, trim extremes)")

    if auto_clean:
        df_clean = integrated_df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                try:
                    converted = pd.to_numeric(df_clean[col], errors="coerce")
                    if converted.notna().sum() > 0:
                        df_clean[col] = converted
                except Exception:
                    continue

        for col in df_clean.columns:
            if "age" in col.lower() and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].clip(lower=0, upper=120)

        for col in df_clean.columns:
            if ("income" in col.lower() or "salary" in col.lower()) and pd.api.types.is_numeric_dtype(
                df_clean[col]
            ):
                df_clean[col] = df_clean[col].clip(lower=0, upper=1_000_000)

        cleaned_df = df_clean
    else:
        cleaned_df = integrated_df

    st.markdown("### 6️⃣ Cleaned / Integrated Dataset Preview")
    st.dataframe(cleaned_df.head(100))
    st.write("Shape:", cleaned_df.shape)

    st.markdown("### 7️⃣ Download Cleaned Dataset")
    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download cleaned data as CSV",
        data=csv_bytes,
        file_name="cleaned_integrated_data.csv",
        mime="text/csv",
        key="download_cleaned",
    )

# --- EDA TAB ---
with tab_eda:
    st.subheader("8️⃣ Exploratory Data Analysis (EDA)")
    if "cleaned_df" in locals():
        df_for_eda = cleaned_df
    else:
        # fallback: attempt from session state or stop
        st.info("Cleaned dataset not found. Run integration & cleaning first (Data Quality tab).")
        st.stop()

    show_basic_eda(df_for_eda)

# --- ML ANOMALIES TAB ---
with tab_ml:
    st.subheader("9️⃣ Optional: ML-based Anomaly Detection")

    if "cleaned_df" not in locals():
        st.info("Cleaned dataset not found. Run integration & cleaning first (Data Quality tab).")
        st.stop()

    run_ml = st.checkbox("Run Isolation Forest anomaly detection on numeric variables")

    if run_ml:
        contamination = st.slider("Anomaly proportion (contamination)", 0.01, 0.2, 0.05, 0.01)
        anomaly_flag = run_isolation_forest(cleaned_df, contamination=contamination)
        if anomaly_flag.sum() > 0:
            st.warning(f"Isolation Forest flagged {int(anomaly_flag.sum())} potential anomalous records.")
            st.dataframe(cleaned_df[anomaly_flag == 1].head(50))
        else:
            st.write("No anomalies flagged (or model not run).")

st.markdown("---")
st.caption(
    "All cleaning steps are transparent and reproducible. "
    "Use the built-in samples, synthetic demo data, or your own files to test the agent."
)
