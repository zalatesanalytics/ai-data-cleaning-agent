# utils/cleaning.py
import pandas as pd
import numpy as np

def basic_cleaning_report(df: pd.DataFrame):
    report = {}

    # Missing values
    missing = df.isna().sum()
    report["missing_counts"] = missing.to_dict()

    # Numeric columns: outliers (simple 1st pass using IQR)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        outlier_info[col] = int(outliers)
    report["outliers"] = outlier_info

    # Simple logical checks: age and income if present
    logical_issues = {}
    if "age" in df.columns:
        logical_issues["age_negative_or_too_high"] = int(((df["age"] < 0) | (df["age"] > 120)).sum())

    for inc_col in [c for c in df.columns if "inc" in c.lower() or "income" in c.lower()]:
        logical_issues[f"{inc_col}_negative"] = int((df[inc_col] < 0).sum())

    report["logical_issues"] = logical_issues

    return report

def apply_basic_cleaning(df: pd.DataFrame):
    df_clean = df.copy()

    # Try convert numeric-like columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            # Try to coerce to numeric where possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    # Example cleaning rules:
    # - Clip age between 0 and 120 if numeric
    if "age" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["age"]):
        df_clean.loc[df_clean["age"] < 0, "age"] = np.nan
        df_clean.loc[df_clean["age"] > 120, "age"] = np.nan

    # - Replace negative incomes with NaN
    for inc_col in [c for c in df_clean.columns if "inc" in c.lower() or "income" in c.lower()]:
        if pd.api.types.is_numeric_dtype(df_clean[inc_col]):
            df_clean.loc[df_clean[inc_col] < 0, inc_col] = np.nan

    # You can expand rules as needed.
    return df_clean
