"""
AI Data Cleaning & Integration Agent

Core Responsibilities
- Ingest multiple datasets (CSV, Excel, Stata, SPSS, JSON, pickle).
- Detect and correct inconsistencies across numeric, categorical, and datetime variables.
- Harmonize semantically similar variables across datasets (e.g., inc, income1, income2, household_income).
- Support multi-dataset merging or appending.
- Perform automatic cleaning by default, but keep the user in full control.

Behavior
- Automatic integration + cleaning is the default path.
- After cleaning, the agent shows BEFORE vs AFTER diagnostics.
- Then it asks the user if they are satisfied with the automatic cleaning.
  - If YES → the cleaned dataset is used for EDA and download.
  - If NO → the integrated but uncleaned dataset is used (manual cleaning mode).
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports for SPSS
try:
    import pyreadstat  # for .sav
except ImportError:
    pyreadstat = None


# =========================================
# 1. Synthetic example dataset
# =========================================

def generate_synthetic_dataset(n: int = 300) -> pd.DataFrame:
    """Generate a synthetic dataset with missing values, extremes, and inconsistencies."""
    rng = np.random.default_rng(42)
    ages = rng.integers(-5, 150, size=n)  # includes impossible ages
    income = rng.normal(3000, 1500, size=n)
    income[rng.choice(n, size=15, replace=False)] = -100   # negative income
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
# 2. File loading
# =========================================

def load_file(uploaded_file) -> pd.DataFrame:
    """Load a file into a pandas DataFrame based on extension."""
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


# =========================================
# 3. Schema & name normalization
# =========================================

def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Standardize column names to snake_case, but keep original names in a mapping.
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
    - simple synonyms for common variables
    Returns tuples of (file_name, col_in_that_file, matching_col_from_some_other_file).
    """
    import difflib

    col_entries = []  # (file, col)
    for fname, df in dfs.items():
        for col in df.columns:
            col_entries.append((fname, col))

    suggestions: List[Tuple[str, str, str]] = []
    synonyms = {
        "sex": ["sex", "gender", "sexe"],
        "gender": ["gender", "sex", "sexe"],
        "age": ["age", "age_years", "years", "age_yrs", "age_child", "age2"],
        "id": ["id", "hhid", "household_id", "respondent_id"],
        "income": ["inc", "income", "income1", "income2", "income3", "income_monthly",
                   "household_income", "hhinc"],
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

            syn_match = False
            for _, variants in synonyms.items():
                if c1.lower() in variants and c2.lower() in variants:
                    syn_match = True
                    break

            if syn_match or ratio > 0.8:
                suggestions.append((f1, c1, c2))

    unique_suggestions = list(dict.fromkeys(suggestions))
    return unique_suggestions


# =========================================
# 4. Data cleaning utilities
# =========================================

PLACEHOLDER_VALUES = {"?", "NA", "N/A", "999", "9999", "Unknown", "unknown"}


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic structural cleaning: strip strings, remove empty rows/cols, reset index."""
    df = df.copy()

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
    """
    Detect simple logical inconsistencies:
    - adolescent age and higher education
    - employed == "Yes" and income <= 0 or missing
    """
    messages: List[str] = []

    # Age
    age_col = None
    age_series = None
    for c in df.columns:
        if "age" in c.lower():
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                age_col = c
                age_series = s
                break

    # Education
    edu_col = None
    edu_keywords = ["educ", "school", "grade", "class", "level"]
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in edu_keywords):
            edu_col = c
            break

    if age_col is not None and edu_col is not None:
        edu_series = df[edu_col].astype(str).str.lower()
        higher_terms = [
            "university",
            "college",
            "bachelor",
            "master",
            "phd",
            "higher",
            "diploma",
            "degree",
        ]
        higher_mask = edu_series.str.contains("|".join(higher_terms), na=False)
        young_mask = (age_series >= 5) & (age_series < 18)
        count = int((young_mask & higher_mask).sum())
        if count > 0:
            messages.append(
                f"{count} records: age 5–17 but education suggests higher/tertiary level "
                f"(column '{edu_col}')."
            )

    # Employment & income
    emp_col = None
    income_col = None
    for c in df.columns:
        lc = c.lower()
        if emp_col is None and lc in {"employed", "employment_status", "employment", "work_status"}:
            emp_col = c
        if income_col is None and any(k in lc for k in ["income", "salary", "wage", "earning", "pay"]):
            income_col = c

    if emp_col is not None and income_col is not None:
        emp_series = df[emp_col].astype(str).str.lower()
        inc_series = pd.to_numeric(df[income_col], errors="coerce")
        with np.errstate(invalid="ignore"):
            mask = (emp_series == "yes") & ((inc_series <= 0) | inc_series.isna())
        count = int(mask.sum())
        if count > 0:
            messages.append(
                f"{count} records: employed='Yes' but income is 0 or missing "
                f"(columns '{emp_col}' and '{income_col}')."
            )

    return messages


def auto_fix_age_education_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Fix cases where age is in 5–17 band but education is unrealistically high."""
    df = df.copy()

    age_col = None
    age_series = None
    for c in df.columns:
        if "age" in c.lower():
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                age_col = c
                age_series = s
                break

    edu_col = None
    edu_keywords = ["educ", "school", "grade", "class", "level"]
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in edu_keywords):
            edu_col = c
            break

    if age_col is None or edu_col is None:
        return df

    edu_series = df[edu_col].astype(str)
    band_mask = (age_series >= 5) & (age_series < 18)
    higher_terms = [
        "university",
        "college",
        "bachelor",
        "master",
        "phd",
        "higher",
        "diploma",
        "degree",
    ]
    higher_mask = edu_series.str.lower().str.contains("|".join(higher_terms), na=False)

    inconsistent_mask = band_mask & higher_mask
    if inconsistent_mask.sum() == 0:
        return df

    normal_band_mask = band_mask & ~inconsistent_mask
    if normal_band_mask.sum() > 0:
        mode_vals = edu_series[normal_band_mask].mode()
        if len(mode_vals) > 0:
            replacement = mode_vals.iloc[0]
        else:
            replacement = "Secondary"
    else:
        replacement = "Secondary"

    df.loc[inconsistent_mask, edu_col] = replacement
    return df


def auto_fix_employment_income_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """For employed='Yes' with income <=0 or missing, impute income using median income among employed."""
    df = df.copy()

    emp_col = None
    income_col = None
    for c in df.columns:
        lc = c.lower()
        if emp_col is None and lc in {"employed", "employment_status", "employment", "work_status"}:
            emp_col = c
        if income_col is None and any(k in lc for k in ["income", "salary", "wage", "earning", "pay"]):
            income_col = c

    if emp_col is None or income_col is None:
        return df

    emp_series = df[emp_col].astype(str).str.lower()
    inc_series = pd.to_numeric(df[income_col], errors="coerce")

    employed_mask = emp_series == "yes"
    valid_income_mask = employed_mask & (inc_series > 0)
    if valid_income_mask.sum() == 0:
        return df

    median_income = inc_series[valid_income_mask].median()
    fix_mask = employed_mask & ((inc_series <= 0) | inc_series.isna())
    df.loc[fix_mask, income_col] = median_income
    return df


def strong_numeric_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strong numeric cleaning:
    - Convert numeric-like text to numeric.
    - Age-like: 0–120, impute median, clip.
    - Income-like: 0–1e6, impute median, clip.
    - Others: trim extreme outliers, impute median.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        col_lower = col.lower()

        if "age" in col_lower:
            s = s.where((s >= 0) & (s <= 120), np.nan)
            if s.notna().sum() > 0:
                median_age = s.median()
                s = s.fillna(median_age)
            s = s.clip(lower=0, upper=120)
            df[col] = s

        elif any(k in col_lower for k in ["income", "salary", "wage", "earning", "pay"]):
            s = s.where((s >= 0) & (s <= 1_000_000), np.nan)
            if s.notna().sum() > 0:
                median_inc = s.median()
                s = s.fillna(median_inc)
            s = s.clip(lower=0, upper=1_000_000)
            df[col] = s

        else:
            s_valid = s.dropna()
            if s_valid.empty:
                df[col] = s
                continue
            q1, q3 = np.percentile(s_valid, [25, 75])
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            s = s.where((s >= lower) & (s <= upper), np.nan)
            if s.notna().sum() > 0:
                median_val = s.median()
                s = s.fillna(median_val)
            df[col] = s

    return df


def auto_impute_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing categorical values using the mode."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_vals = df[col].mode(dropna=True)
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
    return df


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
# 6. Semantic variable collapsing (income, age, etc.)
# =========================================

SEMANTIC_GROUPS = {
    "income": [
        "inc", "income", "income1", "income2", "income3",
        "income_monthly", "hhinc", "household_income", "hh_income", "householdinc",
    ],
    "age": [
        "age", "age_yrs", "age_years", "age2", "age_child", "child_age",
    ],
}


def collapse_semantic_groups(df: pd.DataFrame,
                             groups: Dict[str, List[str]] = SEMANTIC_GROUPS
                             ) -> pd.DataFrame:
    """
    For each semantic group (e.g., income), find all related columns and
    collapse them into a single canonical column (first non-null per row),
    then drop the redundant columns.
    """
    df = df.copy()

    for canonical, patterns in groups.items():
        fam_cols = []
        for col in df.columns:
            lc = col.lower()
            if any(p in lc for p in patterns):
                fam_cols.append(col)

        if not fam_cols:
            continue

        if canonical in fam_cols:
            base_col = canonical
            other_cols = [c for c in fam_cols if c != canonical]
            if other_cols:
                df[base_col] = df[[base_col] + other_cols].bfill(axis=1).iloc[:, 0]
        else:
            df[canonical] = df[fam_cols].bfill(axis=1).iloc[:, 0]
            base_col = canonical
            other_cols = fam_cols

        cols_to_drop = [c for c in fam_cols if c != base_col]
        df = df.drop(columns=cols_to_drop)

    df = df.loc[:, ~df.columns.duplicated()]
    return df


# =========================================
# 7. Integration: append vs merge
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
    """Vertically append multiple DataFrames with basic cleaning."""
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
# 8. EDA helpers
# =========================================

def show_basic_eda(df: pd.DataFrame):
    st.markdown("### 📊 Descriptive Statistics")
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        st.write(df.describe(include=[np.number]).T)
    else:
        st.write("No numeric variables for descriptive statistics.")

    st.markdown("### 📋 Categorical Frequencies (Top 10)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        st.write("No categorical columns detected.")
    else:
        for col in cat_cols:
            with st.expander(f"Variable: {col}", expanded=False):
                st.write(df[col].value_counts(dropna=False).head(10))

    st.markdown("### 🔗 Correlation Heatmap (numeric)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
    else:
        st.write("Not enough numeric variables for correlation heatmap.")

    st.markdown("### 📈 Simple Histogram")
    num_cols = list(num_df.columns)
    if num_cols:
        col_to_plot = st.selectbox("Select numeric variable for histogram", num_cols)
        st.bar_chart(num_df[col_to_plot].dropna().value_counts().sort_index())
    else:
        st.write("No numeric columns available for plotting.")


def generate_narrative_from_eda(df: pd.DataFrame, title: str = "EDA summary") -> str:
    """Generate a simple narrative text based on numeric and categorical distributions."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    lines = [f"**Narrative summary – {title}**"]
    lines.append(
        f"- The dataset used for EDA contains **{df.shape[0]} observations** and **{df.shape[1]} variables**."
    )
    lines.append(
        f"- Of these, **{len(numeric_cols)} numeric** variables and **{len(cat_cols)} categorical** variables were included in the analysis."
    )

    if numeric_cols:
        lines.append("")
        lines.append("**Numeric variables – central tendency and spread**")
        desc = df[numeric_cols].describe().T
        for col in numeric_cols[:5]:
            if col in desc.index:
                row = desc.loc[col]
                lines.append(
                    f"- `{col}` has a mean of **{row['mean']:.2f}**, "
                    f"ranging from **{row['min']:.2f}** to **{row['max']:.2f}**."
                )

    if cat_cols:
        lines.append("")
        lines.append("**Categorical variables – dominant categories**")
        for col in cat_cols[:5]:
            vc = df[col].value_counts(dropna=True)
            if not vc.empty:
                top_cat = vc.index[0]
                top_count = int(vc.iloc[0])
                top_pct = (top_count / len(df) * 100) if len(df) > 0 else 0
                lines.append(
                    f"- In `{col}`, the most frequent category is **{top_cat}** "
                    f"({top_count} records, ~{top_pct:.1f}% of observations)."
                )

    lines.append("")
    lines.append(
        "Overall, these patterns provide a starting point for deeper inferential analysis, "
        "segmentation, or model-building, depending on the project objectives."
    )

    return "\n".join(lines)


# =========================================
# 9. Streamlit App
# =========================================

st.set_page_config(page_title="AI Data Cleaning & Integration Agent", layout="wide")

st.title("🧹 AI Data-Cleaning, Integration & Analysis Agent")

st.write(
    """
This agent ingests multiple datasets, detects schema similarities, automatically
integrates and cleans them by default, and then asks you whether you are satisfied
with the result. If not, you can keep the integrated but uncleaned data for manual
cleaning and custom analysis.
"""
)

# --- Sidebar options ---
st.sidebar.header("Data & Options")

use_synthetic = st.sidebar.checkbox("Use synthetic example dataset (test mode)", value=False)

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more datasets",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls", "dta", "sav", "json", "pkl", "pickle", "tsv", "txt"],
)

# Collect raw dfs
dfs_raw: Dict[str, pd.DataFrame] = {}

if use_synthetic:
    dfs_raw["synthetic.csv"] = generate_synthetic_dataset()

if uploaded_files:
    for uf in uploaded_files:
        try:
            df = load_file(uf)
            dfs_raw[uf.name] = df
        except Exception as e:
            st.error(f"❌ Could not read file `{uf.name}`: {e}")

if not dfs_raw:
    st.info("Use the sidebar to upload files and/or enable synthetic data to begin.")
    st.stop()

# 1️⃣ Preview raw datasets
st.subheader("1️⃣ Preview of Uploaded Datasets")
for fname, df in dfs_raw.items():
    with st.expander(f"File: {fname} (shape={df.shape})", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(10))

# Normalize column names
dfs_norm: Dict[str, pd.DataFrame] = {}
col_maps: Dict[str, Dict[str, str]] = {}
for fname, df in dfs_raw.items():
    ndf, cmap = normalize_column_names(df)
    dfs_norm[fname] = ndf
    col_maps[fname] = cmap

# 2️⃣ Schema summary
st.subheader("2️⃣ Schema Summary (normalized names)")
schema_df = get_schema_summary(dfs_norm)
st.dataframe(schema_df)

# 3️⃣ Column similarity & harmonization
st.subheader("3️⃣ Column Similarity & Harmonization")

harmonization_mappings: Dict[str, Dict[str, str]] = {fname: {} for fname in dfs_norm.keys()}

suggestions = suggest_similar_columns(dfs_norm)
if suggestions:
    st.write(
        "The agent suggests that some variables are likely the **same concept** across datasets.\n"
        "By default, they will be harmonized to a common name when you approve.\n"
        "✅ Only mark them as unrelated if you are sure they should remain separate."
    )

    for idx, (f1, c1, c2) in enumerate(suggestions):
        st.markdown(f"- Suggested match: `{c1}` in **{f1}** ↔ `{c2}` in another file")
        keep_separate = st.checkbox(
            "Keep these separate (do NOT harmonize)",
            key=f"suggest_unrelated_{idx}",
        )
        if not keep_separate:
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

if st.button("Apply harmonization"):
    st.session_state["harmonized_dfs"] = harmonize_columns(dfs_norm, harmonization_mappings)
    st.success("Harmonization applied.")
else:
    if "harmonized_dfs" not in st.session_state:
        st.session_state["harmonized_dfs"] = dfs_norm

harmonized_dfs: Dict[str, pd.DataFrame] = st.session_state["harmonized_dfs"]

# 4️⃣ Integration (append/merge) + automatic cleaning (default)
st.subheader("4️⃣ Integrate & Auto-Clean (default)")

integration_mode = st.radio(
    "Integration mode",
    ["Append (stack rows)", "Merge (join on ID)"],
    index=0,  # default to Append
)

integrate_and_clean = st.button("Run integration + automatic cleaning")

if integrate_and_clean:
    # Integration
    if integration_mode == "Append (stack rows)":
        integrated_df = safely_append(harmonized_dfs)
        integrated_df = collapse_semantic_groups(integrated_df)
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
            st.warning("No obvious ID columns found. Type the key manually if you know it.")
        key = st.text_input("Merge key column name:", value=candidate_ids[0] if candidate_ids else "")
        merge_type = st.selectbox("Merge type", ["outer", "inner", "left", "right"], index=0)

        if not key:
            st.error("Please specify a key column for merging.")
            st.stop()

        integrated_df = safely_merge(harmonized_dfs, key=key, how=merge_type)
        if integrated_df.empty:
            st.stop()
        integrated_df = collapse_semantic_groups(integrated_df)
        st.success(f"Merged datasets on `{key}`. Result shape: {integrated_df.shape}")

    # Save integrated df
    st.session_state["integrated_df"] = integrated_df

    # Automatic cleaning (DEFAULT PATH)
    df_clean = integrated_df.copy()
    df_clean = strong_numeric_cleaning(df_clean)
    df_clean = auto_impute_categorical_missing(df_clean)
    df_clean = auto_fix_age_education_inconsistencies(df_clean)
    df_clean = auto_fix_employment_income_inconsistencies(df_clean)

    st.session_state["cleaned_df_auto"] = df_clean

# If integration not yet run, stop here
if "integrated_df" not in st.session_state or "cleaned_df_auto" not in st.session_state:
    st.info("Click 'Run integration + automatic cleaning' to continue.")
    st.stop()

integrated_df = st.session_state["integrated_df"]
cleaned_df_auto = st.session_state["cleaned_df_auto"]

# 5️⃣ Data-quality assessment BEFORE vs AFTER
st.subheader("5️⃣ Data-Quality Assessment – BEFORE vs AFTER Automatic Cleaning")

miss_before = detect_missingness(integrated_df)
miss_after = detect_missingness(cleaned_df_auto)
ext_before = detect_numeric_extremes(integrated_df)
ext_after = detect_numeric_extremes(cleaned_df_auto)
logical_msgs_before = detect_logical_inconsistencies(integrated_df)
logical_msgs_after = detect_logical_inconsistencies(cleaned_df_auto)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Missing values – BEFORE cleaning**")
    st.dataframe(miss_before)
with col2:
    st.markdown("**Missing values – AFTER cleaning**")
    st.dataframe(miss_after)

st.markdown("**Numeric extremes & impossible values – BEFORE cleaning**")
st.dataframe(ext_before)

st.markdown("**Numeric extremes & impossible values – AFTER cleaning**")
st.dataframe(ext_after)

with st.expander("Logical inconsistencies – BEFORE cleaning", expanded=False):
    if logical_msgs_before:
        for m in logical_msgs_before:
            st.warning(m)
    else:
        st.write("No simple logical inconsistencies detected by current rules.")

with st.expander("Logical inconsistencies – AFTER cleaning", expanded=False):
    if logical_msgs_after:
        for m in logical_msgs_after:
            st.warning(m)
    else:
        st.write("No simple logical inconsistencies detected after cleaning rules.")

# 6️⃣ Ask user if they accept automatic cleaning
st.subheader("6️⃣ Do you accept the automatic cleaning result?")

choice = st.radio(
    "Choose which dataset you want to continue with:",
    [
        "✅ Yes – use the automatically cleaned dataset",
        "❌ No – keep the integrated but uncleaned dataset (manual cleaning)",
    ],
    index=0,
)

if choice.startswith("✅"):
    final_df = cleaned_df_auto
    st.success("Using AUTOMATICALLY CLEANED dataset for EDA and download.")
else:
    final_df = integrated_df
    st.warning("Using INTEGRATED BUT UNCLEANED dataset. You can clean it manually or offline.")

st.subheader("7️⃣ Final Dataset Preview")
st.dataframe(final_df.head(100))
st.write("Shape:", final_df.shape)

# Download final dataset
st.subheader("8️⃣ Download Final Dataset")
csv_bytes = final_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download final dataset as CSV",
    data=csv_bytes,
    file_name="final_data_clean_or_raw.csv",
    mime="text/csv",
)

# 9️⃣ Optional ML-based anomaly detection
st.subheader("9️⃣ Optional: ML-based Anomaly Detection (Isolation Forest)")

run_ml = st.checkbox("Run Isolation Forest anomaly detection on numeric variables")

if run_ml:
    contamination = st.slider("Anomaly proportion (contamination)", 0.01, 0.2, 0.05, 0.01)
    anomaly_flag = run_isolation_forest(final_df, contamination=contamination)
    if anomaly_flag.sum() > 0:
        st.warning(f"Isolation Forest flagged {int(anomaly_flag.sum())} potential anomalous records.")
        st.dataframe(final_df[anomaly_flag == 1].head(50))
    else:
        st.write("No anomalies flagged (or model not run).")

# 🔟 EDA on final dataset
st.subheader("🔟 Exploratory Data Analysis (EDA) on Final Dataset")

numeric_cols_all = list(final_df.select_dtypes(include=[np.number]).columns)
cat_cols_all = list(final_df.select_dtypes(include=["object", "category"]).columns)

if not numeric_cols_all and not cat_cols_all:
    st.write("No numeric or categorical variables available for EDA.")
else:
    analysis_scope = st.radio(
        "Variables for analysis",
        [
            "All numeric & categorical variables",
            "Select variables manually",
        ],
    )

    if analysis_scope.startswith("Select"):
        st.markdown("**Select numeric variables (if any)**")
        numeric_selected = st.multiselect(
            "Numeric variables",
            options=numeric_cols_all,
            default=numeric_cols_all,
        )

        st.markdown("**Select categorical variables (if any)**")
        cat_selected = st.multiselect(
            "Categorical variables",
            options=cat_cols_all,
            default=cat_cols_all,
        )
    else:
        numeric_selected = numeric_cols_all
        cat_selected = cat_cols_all

    selected_cols = list(dict.fromkeys(numeric_selected + cat_selected))
    if not selected_cols:
        st.info("No variables selected for EDA.")
    else:
        df_for_eda = final_df[selected_cols].copy()
        tab_results, tab_narrative = st.tabs(["📊 EDA Results", "📝 Narrative summary"])
        with tab_results:
            show_basic_eda(df_for_eda)
        with tab_narrative:
            narrative = generate_narrative_from_eda(df_for_eda, title="selected variables")
            st.markdown(narrative)

st.markdown("---")
st.caption(
    "Automatic cleaning is applied by default, but you always remain in control. "
    "If you are not satisfied with the result, keep the integrated raw data and "
    "apply your own cleaning rules."
)
