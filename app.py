"""
Zalates Analytics – AI Data Cleaning & Integration Agent
(Futuristic but readable theme + GPS maps + rich visualizations)
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px  # for nicer geo maps & charts

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
    Suggest potentially similar columns across files based on simple heuristics.
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
        "income": [
            "inc", "income", "income1", "income2", "income3",
            "income_monthly", "household_income", "hhinc"
        ],
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

PLACEHOLDER_VALUES = {"?", "NA", "N/A", "na", "NaN", "nan", "999", "9999", "Unknown", "unknown"}


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
        if emp_col is None and lc in {
            "employed", "employment_status", "employment", "work_status"
        }:
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
    """For employed='Yes' with income <=0 or missing, impute income using median among employed."""
    df = df.copy()

    emp_col = None
    income_col = None
    for c in df.columns:
        lc = c.lower()
        if emp_col is None and lc in {
            "employed", "employment_status", "employment", "work_status"
        }:
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


def final_post_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass:
    - Replace common placeholder strings with NaN.
    - Drop rows that are completely empty.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(list(PLACEHOLDER_VALUES), np.nan)
    df = df.dropna(how="all")
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
# 9. Streamlit App – layout & branding (new glass dashboard)
# =========================================

st.set_page_config(
    page_title="Zalates Analytics – AI Data Cleaning Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- GLOBAL THEME: light pastel “glass” dashboard ----------
st.markdown(
    """
    <style>
    /* Overall background gradient */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #e0f2fe 0, #f5f3ff 35%, #fdf2f8 70%, #f9fafb 100%);
    }

    /* Main container */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 22px 40px rgba(15,23,42,0.10);
        border: 1px solid rgba(226,232,240,0.8);
    }
    .glass-card h4, .glass-card h5, .glass-card p {
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
        color: #0f172a;
    }

    /* Metric pills (top mini-cards) */
    .metric-card {
        background: linear-gradient(135deg,#eef2ff,#e0f2fe);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        box-shadow: 0 12px 24px rgba(148,163,184,0.3);
        border: 1px solid rgba(191,219,254,0.9);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.70rem;
        color: #10b981;
    }

    /* Top bar */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .top-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }
    .top-subtitle {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Search bar look */
    .search-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.15rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15,23,42,0.96) !important;
        padding: 1.2rem 1rem !important;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #f9fafb !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #020617 !important;
        border-radius: 12px !important;
        border: 1px dashed #4b5563 !important;
    }

    /* Remove excessive gap under radio labels etc. */
    .element-container:has(> div[data-testid="stRadio"]) {
        margin-bottom: 0.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HEADER (logo + title on gradient pill) ----------
logo_path = "logo-png-circle2.png"  # ensure this exists next to app.py
c_logo, c_title = st.columns([1, 5])
with c_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
with c_title:
    st.markdown(
        """
        <div class="glass-card" style="background:linear-gradient(135deg,#4f46e5,#6366f1,#22c55e);color:white;">
            <div class="top-bar">
                <div>
                    <div class="top-title">AI Data Cleaning & Integration Agent</div>
                    <div class="top-subtitle">
                        Zalates Analytics • clean, harmonise & explore multi-source datasets.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- SIDEBAR: inputs & navigation ----------
st.sidebar.title("🧾 Data Inputs")
use_synthetic = st.sidebar.checkbox("Use synthetic example dataset", value=False)

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more datasets",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls", "dta", "sav", "json", "pkl", "pickle", "tsv", "txt"],
)

st.sidebar.caption(
    "Tip: For GPS maps, include `latitude`/`longitude` and, ideally, a food security or risk indicator."
)

# collect raw dfs
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
    st.info("Upload at least one dataset or enable the synthetic example from the sidebar to begin.")
    st.stop()

# ---------- PREVIEW RAW DATA ----------
st.markdown("### 📂 Raw Files Preview")
for fname, df in dfs_raw.items():
    with st.expander(f"📁 {fname}  • shape={df.shape}", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(8))

# normalize names
dfs_norm: Dict[str, pd.DataFrame] = {}
col_maps: Dict[str, Dict[str, str]] = {}
for fname, df in dfs_raw.items():
    ndf, cmap = normalize_column_names(df)
    dfs_norm[fname] = ndf
    col_maps[fname] = cmap

schema_df = get_schema_summary(dfs_norm)

# ---------- HARMONISATION ----------
st.markdown("### 🧬 Schema & Harmonisation")
with st.expander("See normalised schema for all files", expanded=False):
    st.dataframe(schema_df, use_container_width=True)

harmonization_mappings: Dict[str, Dict[str, str]] = {fname: {} for fname in dfs_norm.keys()}
suggestions = suggest_similar_columns(dfs_norm)

if suggestions:
    st.caption(
        "The agent found variables that look semantically similar across files. "
        "Uncheck anything you want to keep separate."
    )
    for idx, (f1, c1, c2) in enumerate(suggestions):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{f1}** → `{c1}`")
        with col2:
            st.write(f"↔ `{c2}` (other file)")
        with col3:
            keep_sep = st.checkbox(
                "Separate",
                key=f"suggest_sep_{idx}",
                value=False,
            )
        if not keep_sep:
            base = c1
            harmonization_mappings[f1][c1] = base
            for fname, df in dfs_norm.items():
                if c2 in df.columns:
                    harmonization_mappings[fname][c2] = base
else:
    st.caption("No strong automatic matches; you can still rename manually below if needed.")

with st.expander("Manual column renaming (optional)", expanded=False):
    for fname, df in dfs_norm.items():
        st.markdown(f"**File: {fname}**")
        for col in df.columns:
            new_name = st.text_input(
                f"Standard name for `{col}` in {fname} (leave blank to keep as is)",
                value="",
                key=f"manual_{fname}_{col}",
            )
            if new_name:
                harmonization_mappings[fname][col] = new_name

if st.button("✅ Apply harmonisation"):
    st.session_state["harmonized_dfs"] = harmonize_columns(dfs_norm, harmonization_mappings)
    st.success("Harmonised variable names applied.")
else:
    if "harmonized_dfs" not in st.session_state:
        st.session_state["harmonized_dfs"] = dfs_norm

harmonized_dfs: Dict[str, pd.DataFrame] = st.session_state["harmonized_dfs"]

# ---------- INTEGRATION & CLEANING ----------
st.markdown("### 🔗 Integration & Automatic Cleaning")

integration_mode = st.radio(
    "How should multiple files be combined?",
    ["Append (stack rows)", "Merge (join on ID)"],
    index=0,
    horizontal=True,
)

run_integration = st.button("🚀 Run integration + cleaning")

if run_integration:
    # integration
    if integration_mode == "Append (stack rows)":
        integrated_df = safely_append(harmonized_dfs)
        integrated_df = collapse_semantic_groups(integrated_df)
        st.success(f"Appended {len(harmonized_dfs)} datasets → integrated shape {integrated_df.shape}")
    else:
        st.info("Trying to guess an ID column (e.g. `id`, `hhid`, `household_id`).")
        candidate_ids = set()
        for df in harmonized_dfs.values():
            for c in df.columns:
                if any(k in c.lower() for k in ["id", "hhid", "household"]):
                    candidate_ids.add(c)
        candidate_ids = sorted(candidate_ids)
        key = st.text_input(
            "Merge key column name",
            value=candidate_ids[0] if candidate_ids else "",
        )
        merge_type = st.selectbox("Merge type", ["outer", "inner", "left", "right"], index=0)
        if not key:
            st.error("Please provide an ID column name for merging.")
            st.stop()
        integrated_df = safely_merge(harmonized_dfs, key=key, how=merge_type)
        if integrated_df.empty:
            st.stop()
        integrated_df = collapse_semantic_groups(integrated_df)
        st.success(f"Merged datasets on `{key}` ({merge_type} join) → shape {integrated_df.shape}")

    st.session_state["integrated_df"] = integrated_df

    # automatic cleaning
    df_clean = integrated_df.copy()
    df_clean = strong_numeric_cleaning(df_clean)
    df_clean = auto_impute_categorical_missing(df_clean)
    df_clean = auto_fix_age_education_inconsistencies(df_clean)
    df_clean = auto_fix_employment_income_inconsistencies(df_clean)
    df_clean = final_post_clean(df_clean)

    st.session_state["cleaned_df_auto"] = df_clean

if "integrated_df" not in st.session_state or "cleaned_df_auto" not in st.session_state:
    st.info("Click **Run integration + cleaning** above to generate the dashboard.")
    st.stop()

integrated_df = st.session_state["integrated_df"]
cleaned_df_auto = st.session_state["cleaned_df_auto"]

# basic quality summaries
miss_before = detect_missingness(integrated_df)
miss_after = detect_missingness(cleaned_df_auto)
ext_before = detect_numeric_extremes(integrated_df)
ext_after = detect_numeric_extremes(cleaned_df_auto)

logical_msgs_before = detect_logical_inconsistencies(integrated_df)
logical_msgs_after = detect_logical_inconsistencies(cleaned_df_auto)

# ---------- choice: cleaned vs raw ----------
st.markdown("### 🧪 Which dataset should power the dashboard?")
choice = st.radio(
    "",
    [
        "Use automatically cleaned dataset (recommended)",
        "Use integrated but uncleaned dataset",
    ],
    index=0,
)
if choice.startswith("Use automatically"):
    final_df = cleaned_df_auto
else:
    final_df = integrated_df

numeric_cols_all = list(final_df.select_dtypes(include=[np.number]).columns)
cat_cols_all = list(final_df.select_dtypes(include=["object", "category"]).columns)

# ========== SIDEBAR NAV FOR PAGES ==========
st.sidebar.title("📊 Analytics Workspace")
page = st.sidebar.radio(
    "View",
    ["Dashboard", "Summary & EDA", "Visualizations", "Slicer / Cross-tabs", "Narrative summary"],
)

st.markdown("---")

# =========================================================
# PAGE 1 – MAIN DASHBOARD (styled like attached design)
# =========================================================
if page == "Dashboard":
    # ---- Top search + small profile card ----
    c1, c2 = st.columns([3, 1.2])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="search-label">Search</div>', unsafe_allow_html=True)
        st.text_input(
            label="",
            value="",
            placeholder="Search variables, columns or notes…",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="glass-card" style="text-align:right;">
                <h5 style="margin:0;font-size:0.85rem;color:#64748b;">Active dataset</h5>
                <p style="margin:0;font-size:1.15rem;font-weight:700;color:#111827;">Zalates Data Lab</p>
                <p style="margin:0.15rem 0 0;font-size:0.75rem;color:#22c55e;">Cleaned & ready ✅</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Metric row (like "My Portfolio") ----
    rows = final_df.shape[0]
    cols = final_df.shape[1]
    total_cells = rows * cols if rows and cols else 0
    miss_cells = int(final_df.isna().sum().sum())
    miss_pct_overall = (miss_cells / total_cells * 100) if total_cells else 0

    anomalies_possible = ext_after["outliers_iqr"].sum() if not ext_after.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Records</div>
              <div class="metric-value">{rows:,}</div>
              <div class="metric-sub">rows loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Variables</div>
              <div class="metric-value">{cols}</div>
              <div class="metric-sub">columns after harmonisation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Missing cells</div>
              <div class="metric-value">{miss_pct_overall:.1f}%</div>
              <div class="metric-sub">{miss_cells:,} total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Flagged extremes</div>
              <div class="metric-value">{int(anomalies_possible)}</div>
              <div class="metric-sub">IQR / rule-based outliers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Three-panel layout like the design: left (summaries), mid (charts), right (cards) ----
    left_col, mid_col, right_col = st.columns([2.1, 3.1, 2.2])

    # LEFT: “Swaps & Summaries” style – quality tables
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Swaps & Summaries")
        st.caption("Quick view of missingness and extremes for key variables.")
        if not miss_after.empty:
            st.dataframe(miss_after.head(8), use_container_width=True, height=270)
        else:
            st.write("No missingness detected.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:0.75rem;">', unsafe_allow_html=True)
        st.markdown("#### Dataset Snapshot")
        st.dataframe(final_df.head(6), use_container_width=True, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

    # MIDDLE: portfolio-style charts – bar for missingness + numeric dist
    with mid_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Data Quality Status")

        # missingness bar chart
        if not miss_after.empty:
            miss_plot = miss_after.reset_index().rename(columns={"index": "variable"})
            miss_plot = miss_plot.sort_values("missing_pct", ascending=False).head(8)
            fig_miss = px.bar(
                miss_plot,
                x="variable",
                y="missing_pct",
                title="Missingness by variable (%)",
            )
            fig_miss.update_layout(margin=dict(l=5, r=5, t=40, b=40))
            st.plotly_chart(fig_miss, use_container_width=True, height=260)
        else:
            st.info("No missingness summary available.")

        # numeric distribution
        if numeric_cols_all:
            num_for_dash = st.selectbox(
                "Numeric metric to inspect",
                numeric_cols_all,
                key="dash_num",
            )
            fig_hist = px.histogram(
                final_df,
                x=num_for_dash,
                nbins=30,
                title=f"Distribution of {num_for_dash}",
            )
            fig_hist.update_layout(margin=dict(l=5, r=5, t=40, b=40))
            st.plotly_chart(fig_hist, use_container_width=True, height=260)
        else:
            st.info("No numeric variables found for distribution plot.")
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: “Your cards / recent payments” style – files & download
    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Your Datasets")
        st.caption("Files currently integrated in this session.")
        for fname, df in harmonized_dfs.items():
            st.markdown(f"• **{fname}**  \n&nbsp;&nbsp;{df.shape[0]} rows • {df.shape[1]} cols")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:0.75rem;">', unsafe_allow_html=True)
        st.markdown("#### Export Clean Data")
        csv_bytes = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download final dataset (CSV)",
            data=csv_bytes,
            file_name="final_data_clean_or_raw.csv",
            mime="text/csv",
        )
        st.caption("This is the dataset powering all charts above.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:0.75rem;">', unsafe_allow_html=True)
        st.markdown("#### Logical Checks")
        if logical_msgs_after:
            for msg in logical_msgs_after:
                st.markdown(f"- {msg}")
        else:
            st.caption("No rule-based logical inconsistencies detected after cleaning.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE 2 – SUMMARY & EDA
# =========================================================
elif page == "Summary & EDA":
    st.subheader("📋 Summary & EDA (descriptive statistics)")

    if numeric_cols_all:
        st.markdown("**Numeric variables – descriptive statistics**")
        st.dataframe(final_df[numeric_cols_all].describe().T, use_container_width=True)
    else:
        st.info("No numeric variables for descriptive statistics.")

    if cat_cols_all:
        st.markdown("**Categorical variables – top categories**")
        for col in cat_cols_all[:10]:
            with st.expander(f"Variable: {col}", expanded=False):
                st.write(final_df[col].value_counts(dropna=True).head(15))
    else:
        st.info("No categorical variables detected.")

    if len(numeric_cols_all) > 1:
        st.markdown("**Correlation matrix (numeric only)**")
        corr = final_df[numeric_cols_all].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)
    else:
        st.info("Not enough numeric variables for correlation matrix.")

# =========================================================
# PAGE 3 – VISUALIZATIONS
# =========================================================
elif page == "Visualizations":
    st.subheader("📊 Custom Visualizations")

    # numeric plots
    st.markdown("### Numeric variables")
    if numeric_cols_all:
        num_var = st.selectbox("Choose numeric variable", numeric_cols_all, key="viz_num")
        series = final_df[num_var].dropna()
        if not series.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(series, nbins=30, labels={"value": num_var})
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                sorted_series = series.sort_values().reset_index(drop=True)
                fig_line = px.line(sorted_series, labels={"index": "sorted index", "value": num_var})
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Selected numeric variable has only missing values.")
    else:
        st.info("No numeric variables available.")

    # categorical plots
    st.markdown("### Categorical variables")
    if cat_cols_all:
        cat_var = st.selectbox("Choose categorical variable", cat_cols_all, key="viz_cat")
        cat_counts = final_df[cat_var].value_counts(dropna=True).head(15)
        if not cat_counts.empty:
            cat_df = cat_counts.reset_index()
            cat_df.columns = [cat_var, "count"]
            c1, c2 = st.columns(2)
            with c1:
                fig_bar = px.bar(cat_df, x=cat_var, y="count", title=f"{cat_var} counts (top 15)")
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                fig_bar_h = px.bar(cat_df, x="count", y=cat_var, orientation="h")
                st.plotly_chart(fig_bar_h, use_container_width=True)
            fig_pie = px.pie(cat_df, names=cat_var, values="count", title=f"{cat_var} share")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Selected categorical variable has only missing values.")
    else:
        st.info("No categorical variables found.")

    # GPS map
    st.markdown("### 🌍 GPS / Map (if lat/lon available)")
    lat_cols = [c for c in final_df.columns if "lat" in c.lower()]
    lon_cols = [c for c in final_df.columns if "lon" in c.lower() or "lng" in c.lower()]
    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        map_df = final_df[[lat_col, lon_col]].dropna()
        map_df = map_df.rename(columns={lat_col: "lat", lon_col: "lon"})
        if not map_df.empty:
            st.map(map_df, use_container_width=True)
        else:
            st.caption("Latitude/longitude detected but all rows are missing.")
    else:
        st.caption("No latitude/longitude columns detected for mapping.")

# =========================================================
# PAGE 4 – SLICER / CROSS-TABS
# =========================================================
elif page == "Slicer / Cross-tabs":
    st.subheader("🧩 Slicer / Cross-tab Dashboards")

    if not numeric_cols_all or not cat_cols_all:
        st.info("Need at least one numeric and one categorical variable for slicer analysis.")
    else:
        target = st.selectbox("Numeric outcome", numeric_cols_all, key="slicer_target")
        slicer_1 = st.selectbox("First slicer (categorical)", cat_cols_all, key="slicer1")
        slicer_2 = st.selectbox("Second slicer (optional)", ["(none)"] + cat_cols_all, key="slicer2")

        group_cols = [slicer_1] if slicer_2 == "(none)" else [slicer_1, slicer_2]
        grouped = (
            final_df[group_cols + [target]]
            .dropna(subset=[target])
            .groupby(group_cols)[target]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped.rename(columns={"count": "n", "mean": f"{target}_mean"}, inplace=True)

        st.markdown("**Summary table**")
        st.dataframe(grouped, use_container_width=True)

        if slicer_2 == "(none)":
            fig = px.bar(
                grouped,
                x=slicer_1,
                y=f"{target}_mean",
                title=f"Mean {target} by {slicer_1}",
            )
        else:
            fig = px.bar(
                grouped,
                x=slicer_1,
                y=f"{target}_mean",
                color=slicer_2,
                barmode="group",
                title=f"Mean {target} by {slicer_1} and {slicer_2}",
            )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 5 – NARRATIVE SUMMARY
# =========================================================
elif page == "Narrative summary":
    st.subheader("📝 Narrative summary of dataset")
    narrative = generate_narrative_from_eda(final_df, title="full dataset")
    st.markdown(narrative)

st.markdown("---")
st.caption(
    "Dashboard powered by Zalates Analytics • automatic cleaning, harmonisation and visual EDA for multi-source datasets."
)
