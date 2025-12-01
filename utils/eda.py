# utils/eda.py
import pandas as pd

def generate_eda_summary(df: pd.DataFrame):
    summary = {}
    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    summary["columns"] = df.dtypes.astype(str).to_dict()
    summary["head"] = df.head().to_dict(orient="list")

    # Basic numeric stats
    desc = df.describe(include="all").transpose()
    summary["describe"] = desc.to_dict(orient="index")

    return summary
