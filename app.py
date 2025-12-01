Love this direction — let’s make the “automatic cleaning” really *aggressive* so you don’t see negative ages, negative incomes, or uncorrected inconsistencies after cleaning.

Below is a **ready-to-paste full `app.py`** with a much stronger cleaning pipeline:

* **Strong numeric cleaning**:

  * Converts numeric-like text to numbers.
  * For `age*` columns: invalid (<0 or >120) → `NaN`, then impute median age; final clip to `[0,120]`.
  * For income-like columns (`income`, `salary`, `wage`, `earning`, `pay`): invalid (<0 or >1,000,000) → `NaN`, then impute median; final clip to `[0,1,000,000]`.
  * For other numeric columns: trim extreme outliers beyond `Q1 ± 3*IQR` → `NaN`, then impute median.
* **Categorical cleaning**:

  * Impute missing categorical values with mode.
* **Logical fixes**:

  * Age < 18 + “university/college/degree” → education reset to dominant level for that age band (e.g. secondary).
  * Employed = “Yes” + income ≤ 0 or missing → income set to median positive income among employed.

So after **Automatic cleaning**, you should *not* see negative ages/incomes, and inconsistencies should be heavily reduced.

---

```python
import io
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

# We'll import sklearn lazily inside functions to avoid hard failure if missing


# =========================================
# 1. Utility: synthetic example dataset
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

        # Replace placeholder values with NaN
        df[col] = df[col].replace(list(PLACEHOLDER_VALUES), np.nan)

    # Drop fully empty rows/columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Reset index & ensure unique columns
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
    """
    Flag potential extreme values for numeric columns.
    Uses generic IQR + some domain rules for age/income-like columns.
    """
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
    Detect simple logical inconsistencies such as:
    - adolescent age and higher education (university/college/bachelor/master/etc.)
    - employed == "Yes" and income <= 0 or missing
    Works robustly with varied column names and mixed types.
    """
    messages: List[str] = []

    # ---------- Find an age column ----------
    age_col = None
    age_series = None

    for c in df.columns:
        if "age" in c.lower():
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                age_col = c
                age_series = s
                break  # take the first reasonable age column

    # ---------- Find an education column ----------
    edu_col = None
    edu_keywords = ["educ", "school", "grade", "class", "level"]

    for c in df.columns:
        name = c.lower()
        if any(k in name for k in edu_keywords):
            edu_col = c
            break

    # ---------- Rule 1: Adolescent age with higher education ----------
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
        # adolescent band: 5–17
        young_mask = (age_series >= 5) & (age_series < 18)

        count = int((young_mask & higher_mask).sum())
        if count > 0:
            messages.append(
                f"{count} records: age 5–17 but education suggests higher/tertiary level "
                f"(column '{edu_col}')."
            )

    # ---------- Find employment and income columns ----------
    emp_col = None
    income_col = None

    for c in df.columns:
        lc = c.lower()
        if emp_col is None and lc in {"employed", "employment_status", "employment", "work_status"}:
            emp_col = c
        if income_col is None and any(k in lc for k in ["income", "salary", "wage", "earning", "pay"]):
            income_col = c

    # ---------- Rule 2: Employed == Yes but income <= 0 or missing ----------
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
    """
    Fix cases where age is in 5–17 band but education is unrealistically high.
    We reassign education to the most common realistic level observed in that age band.
    """
    df = df.copy()

    # Find age column
    age_col = None
    age_series = None
    for c in df.columns:
        if "age" in c.lower():
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                age_col = c
                age_series = s
                break

    # Find education column
    edu_col = None
    edu_keywords = ["educ", "school", "grade", "class", "level"]
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in edu_keywords):
            edu_col = c
            break

    if age_col is None or edu_col is None:
        return df  # nothing to fix

    edu_series = df[edu_col].astype(str)

    # Define adolescent band and "higher" education markers
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

    # Normal (non-inconsistent) adolescents for reference
    normal_band_mask = band_mask & ~inconsistent_mask
    if normal_band_mask.sum() > 0:
        # Use mode of education among 5–17 non-inconsistent cases
        mode_vals = edu_series[normal_band_mask].mode()
        if len(mode_vals) > 0:
            replacement = mode_vals.iloc[0]
        else:
            replacement = "Secondary"
    else:
        # Fallback if no normal records exist in the band
        replacement = "Secondary"

    df.loc[inconsistent_mask, edu_col] = replacement
    return df


def auto_fix_employment_income_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For employed='Yes' with income <=0 or missing, impute income using median
    positive income among employed.
    """
    df = df.copy()

    # Detect columns as in detect_logical_inconsistencies
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

    # Median positive income among employed
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
    - Converts numeric-like text to numeric.
    - Age-like columns: remove values <0 or >120, then impute median, clip [0,120].
    - Income-like columns: remove values <0 or >1,000,000, then impute median, clip [0,1,000,000].
    - Other numeric columns: trim extreme outliers (Q1±3*IQR), then impute median.
    """
    df = df.copy()

    for col in df.columns:
        # Ensure numeric where possible
        if df[col].dtype == "object":
            # Try convert to numeric; if it fails for all, we keep as object
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        col_lower = col.lower()
        if "age" in col_lower:
            # Age: 0–120
            s = s.where((s >= 0) & (s <= 120), np.nan)
            if s.notna().sum() > 0:
                median_age = s.median()
                s = s.fillna(median_age)
            s = s.clip(lower=0, upper=120)
            df[col] = s

        elif any(k in col_lower for k in ["income", "salary", "wage", "earning", "pay"]):
            # Income: 0–1e6
            s = s.where((s >= 0) & (s <= 1_000_000), np.nan)
            if s.notna().sum() > 0:
                median_inc = s.median()
                s = s.fillna(median_inc)
            s = s.clip(lower=0, upper=1_000_000)
            df[col] = s

        else:
            # Generic numeric: trim extreme outliers and impute
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
    """
    Impute missing categorical values using the mode of each column.
    """
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
    # Convert to full-index series
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
    """
    Apply user-defined mappings: mappings[file][old_col] = new_col.
    """
    out: Dict[str, pd.DataFrame] = {}
    for fname, df in dfs.items():
        df = df.copy()
        if fname in mappings:
            df = df.rename(columns=mappings[fname])
        # Ensure uniqueness after rename
        df = df.loc[:, ~df.columns.duplicated()]
        out[fname] = df
    return out


def safely_append(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Vertically append multiple DataFrames with safety checks.

    When column names are harmonized (e.g., 'educ' and 'education_head'
    mapped to the same standard name), rows from different datasets
    will be appended under that common column, while unique columns
    from each dataset are preserved.
    """
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
        for col in numeric_cols[:5]:  # summarize up to 5 numeric variables
            if col in desc.index:
                row = desc.loc[col]
                lines.append(
                    f"- `{col}` has a mean of **{row['mean']:.2f}**, "
                    f"ranging approximately from **{row['min']:.2f}** to **{row['max']:.2f}**."
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
# 8. Streamlit App
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

# --- Sidebar options ---
st.sidebar.header("Data & Options")

use_synthetic = st.sidebar.checkbox("Use synthetic example dataset", value=False)

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more datasets",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls", "dta", "sav", "json", "pkl", "pickle", "tsv", "txt"],
)

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

st.subheader("1️⃣ Preview of Uploaded Datasets")
for fname, df in dfs_raw.items():
    with st.expander(f"File: {fname} (shape={df.shape})", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(10))

# Normalize columns for internal use (but still show originals above)
dfs_norm: Dict[str, pd.DataFrame] = {}
col_maps: Dict[str, Dict[str, str]] = {}
for fname, df in dfs_raw.items():
    ndf, cmap = normalize_column_names(df)
    dfs_norm[fname] = ndf
    col_maps[fname] = cmap

# Schema summary
st.subheader("2️⃣ Schema Summary")
schema_df = get_schema_summary(dfs_norm)
st.dataframe(schema_df)

# -----------------------------------------
# 3️⃣ Column Similarity & Harmonization
# -----------------------------------------
st.subheader("3️⃣ Column Similarity & Harmonization")

# Initialize mappings
harmonization_mappings: Dict[str, Dict[str, str]] = {fname: {} for fname in dfs_norm.keys()}

suggestions = suggest_similar_columns(dfs_norm)
if suggestions:
    st.write(
        "The agent suggests that some variables are likely the **same concept** across datasets.\n"
        "By default, they will be harmonized to a common name and then appended as a single column.\n"
        "✅ **Check the box only if they are actually unrelated and should NOT be harmonized.**"
    )

    # For each suggestion, we harmonize unless the user marks it as unrelated
    for idx, (f1, c1, c2) in enumerate(suggestions):
        st.markdown(f"- Suggested match: `{c1}` in **{f1}** ↔ `{c2}` in another file")
        not_related = st.checkbox(
            "These are unrelated – keep them separate",
            key=f"suggest_unrelated_{idx}",
        )
        if not not_related:
            base = c1  # choose the first as canonical name
            # Map both columns to same base name
            harmonization_mappings[f1][c1] = base
            for fname, df in dfs_norm.items():
                if c2 in df.columns:
                    harmonization_mappings[fname][c2] = base
else:
    st.write("No strong column similarity suggestions found. You can still harmonize manually.")

# Manual harmonization (always available)
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

# Apply harmonization button
if st.button("Apply harmonization & proceed to integration"):
    st.session_state["harmonized_dfs"] = harmonize_columns(dfs_norm, harmonization_mappings)
    st.success("Harmonization applied.")
else:
    if "harmonized_dfs" not in st.session_state:
        st.session_state["harmonized_dfs"] = dfs_norm

harmonized_dfs: Dict[str, pd.DataFrame] = st.session_state["harmonized_dfs"]

# -----------------------------------------
# 4️⃣ Choose Integration Mode
# -----------------------------------------
st.subheader("4️⃣ Choose Integration Mode")

integration_mode = st.radio(
    "How should the datasets be integrated?",
    ["Append (stack rows)", "Merge (join on ID)"],
)

integrated_df = pd.DataFrame()

if integration_mode == "Append (stack rows)":
    if st.button("Run append integration"):
        integrated_df = safely_append(harmonized_dfs)
        st.success(f"Appended {len(harmonized_dfs)} datasets. Result shape: {integrated_df.shape}")
else:
    # Propose ID candidates
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

    if st.button("Run merge integration"):
        if not key:
            st.error("Please specify a key column for merging.")
        else:
            integrated_df = safely_merge(harmonized_dfs, key=key, how=merge_type)
            if not integrated_df.empty:
                st.success(f"Merged datasets on `{key}`. Result shape: {integrated_df.shape}")

if integrated_df.empty:
    st.stop()

# =========================================
# 5️⃣ Data-quality assessment & cleaning
# =========================================

st.subheader("5️⃣ Data-Quality Assessment – BEFORE Cleaning")

# Before-cleaning diagnostics
miss_before = detect_missingness(integrated_df)
ext_before = detect_numeric_extremes(integrated_df)
logical_msgs_before = detect_logical_inconsistencies(integrated_df)

with st.expander("Missing values summary (before cleaning)", expanded=True):
    st.dataframe(miss_before)

with st.expander("Numeric extremes & impossible values (before cleaning)", expanded=True):
    st.dataframe(ext_before)

with st.expander("Duplicate IDs (before cleaning)", expanded=False):
    candidate_ids = [c for c in integrated_df.columns if "id" in c.lower()]
    if candidate_ids:
        chosen_ids = st.multiselect(
            "Select ID columns to check for duplicates", candidate_ids, default=candidate_ids
        )
        dup_info_before = detect_duplicates(integrated_df, chosen_ids)
        if dup_info_before:
            st.write(dup_info_before)
        else:
            st.write("No duplicates detected for selected ID columns.")
    else:
        st.write("No ID-like columns found for duplicate checks.")

with st.expander("Logical inconsistencies (before cleaning)", expanded=False):
    if logical_msgs_before:
        for m in logical_msgs_before:
            st.warning(m)
    else:
        st.write("No simple logical inconsistencies detected by current rules.")

st.markdown(
    """
**Summary:** These are the potential data-cleaning issues detected (missing data, outliers, duplicates, inconsistencies).  
Now choose whether you want the agent to apply automatic corrections or keep everything manual.
"""
)

cleaning_mode = st.radio(
    "Choose data-cleaning mode",
    [
        "Manual review only (no automatic fixes)",
        "Automatic cleaning (strong numeric & categorical cleaning + logical fixes)",
    ],
)

# Apply cleaning based on mode
if cleaning_mode.startswith("Automatic"):
    df_clean = integrated_df.copy()

    # Strong numeric cleaning (conversions, bounds, imputation)
    df_clean = strong_numeric_cleaning(df_clean)

    # Categorical imputation
    df_clean = auto_impute_categorical_missing(df_clean)

    # Logical fixes: age–education inconsistencies
    df_clean = auto_fix_age_education_inconsistencies(df_clean)

    # Logical fixes: employment–income inconsistencies
    df_clean = auto_fix_employment_income_inconsistencies(df_clean)

    cleaned_df = df_clean
else:
    cleaned_df = integrated_df

st.subheader("6️⃣ Data-quality Assessment – AFTER Cleaning")

miss_after = detect_missingness(cleaned_df)
ext_after = detect_numeric_extremes(cleaned_df)
logical_msgs_after = detect_logical_inconsistencies(cleaned_df)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Missing values – before cleaning**")
    st.dataframe(miss_before)
with col2:
    st.markdown("**Missing values – after cleaning**")
    st.dataframe(miss_after)

st.markdown("**Numeric extremes & impossible values – after cleaning**")
st.dataframe(ext_after)

with st.expander("Logical inconsistencies – after cleaning", expanded=False):
    if logical_msgs_after:
        for m in logical_msgs_after:
            st.warning(m)
    else:
        st.write("No simple logical inconsistencies detected after cleaning rules.")

st.subheader("7️⃣ Cleaned / Integrated Dataset Preview")
st.dataframe(cleaned_df.head(100))
st.write("Shape:", cleaned_df.shape)

# Download
st.subheader("8️⃣ Download Cleaned Dataset")
csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download cleaned data as CSV",
    data=csv_bytes,
    file_name="cleaned_integrated_data.csv",
    mime="text/csv",
)

# =========================================
# 9️⃣ Optional ML-based anomaly detection
# =========================================

st.subheader("9️⃣ Optional: ML-based Anomaly Detection")

run_ml = st.checkbox("Run Isolation Forest anomaly detection on numeric variables")

if run_ml:
    contamination = st.slider("Anomaly proportion (contamination)", 0.01, 0.2, 0.05, 0.01)
    anomaly_flag = run_isolation_forest(cleaned_df, contamination=contamination)
    if anomaly_flag.sum() > 0:
        st.warning(f"Isolation Forest flagged {int(anomaly_flag.sum())} potential anomalous records.")
        st.dataframe(cleaned_df[anomaly_flag == 1].head(50))
    else:
        st.write("No anomalies flagged (or model not run).")

# =========================================
# 🔟 Exploratory Data Analysis (EDA)
# =========================================

st.subheader("🔟 Exploratory Data Analysis (EDA)")

numeric_cols_all = list(cleaned_df.select_dtypes(include=[np.number]).columns)
cat_cols_all = list(cleaned_df.select_dtypes(include=["object", "category"]).columns)

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
        df_for_eda = cleaned_df[selected_cols].copy()

        tab_results, tab_narrative = st.tabs(["📊 EDA Results", "📝 Narrative summary"])

        with tab_results:
            show_basic_eda(df_for_eda)

        with tab_narrative:
            narrative = generate_narrative_from_eda(df_for_eda, title="selected variables")
            st.markdown(narrative)

st.markdown("---")
st.caption(
    "All cleaning steps are transparent and reproducible. "
    "Please review flagged issues and narratives before making final analytical decisions."
)
```

If, after this, you still see negatives or weird ages *in the AFTER tables*, tell me the exact column names that are misbehaving (e.g., `age_yrs`, `hh_income`) and I’ll extend the rules to explicitly catch those patterns too.
