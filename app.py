# app.py / streamlit_app.py

import io
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

import requests  # for KoboToolbox API calls
import plotly.express as px  # interactive charts

# Optional OpenAI import (needed for the openai_available flag)
try:
    import openai
    openai_available = True
except ImportError:
    openai = None
    openai_available = False

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="AI Food & Youth Analysis Assistant",
    layout="wide",
    page_icon="📊",
)

# ---------- Global theming & layout ----------
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 40%, #111827 70%, #451a03 100%);
        color: #f9fafb;
        font-family: "system-ui", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #111827 50%, #0b1120 100%);
        border-right: 1px solid rgba(148,163,184,0.4);
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
        font-size: 0.92rem;
    }
    .hero-card {
        padding: 1.4rem 1.8rem;
        border-radius: 1.4rem;
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 60%, #7c2d12 100%);
        border: 1px solid rgba(251,146,60,0.7);
        box-shadow: 0 22px 60px rgba(15,23,42,0.8);
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(251,146,60,0.25) 0, transparent 70%);
        filter: blur(4px);
        opacity: 0.9;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg,#f97316,#facc15,#f97316);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #e5e7eb;
        opacity: 0.95;
        max-width: 720px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #e5e7eb;
        margin-bottom: 0.4rem;
    }
    .hero-pill span.dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: radial-gradient(circle, #22c55e 0, #16a34a 40%, #166534 100%);
        box-shadow: 0 0 14px rgba(34,197,94,0.8);
    }
    .metric-card {
        border-radius: 1.1rem;
        padding: 0.9rem 1.15rem 0.95rem;
        background: linear-gradient(135deg,rgba(15,23,42,0.96),rgba(30,64,175,0.9));
        border: 1px solid rgba(148,163,184,0.6);
        box-shadow: 0 14px 32px rgba(15,23,42,0.75);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        cursor: default;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 42px rgba(15,23,42,0.9);
        border-color: rgba(251,146,60,0.9);
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #9ca3af;
        margin-bottom: 0.05rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 0;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #e5e7eb;
        opacity: 0.9;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #facc15;
        margin-top: 1.4rem;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        background: rgba(15,23,42,0.95);
        border-radius: 999px;
        padding: 0.25rem;
        border: 1px solid rgba(148,163,184,0.6);
        box-shadow: 0 12px 30px rgba(15,23,42,0.7);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 0.25rem 0.9rem;
        color: #e5e7eb;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg,#fb923c,#f97316);
        color: #111827 !important;
    }
    .dataframe-card {
        background: rgba(15,23,42,0.96);
        border-radius: 1rem;
        border: 1px solid rgba(148,163,184,0.6);
        padding: 0.4rem 0.6rem 0.6rem;
        box-shadow: 0 10px 26px rgba(15,23,42,0.9);
    }
    .stMarkdown, .stText, .stDataFrame {
        color: #e5e7eb;
    }
    /* Narrative cards – darker background and readable text */
    .narrative-card {
        background: rgba(15,23,42,0.96);
        border-radius: 1rem;
        border: 1px solid rgba(148,163,184,0.7);
        padding: 0.9rem 1.1rem;
        box-shadow: 0 12px 32px rgba(15,23,42,0.9);
        color: #f9fafb;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- OPENAI CONFIG -------------------
if openai_available:
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
else:
    openai.api_key = None

if not openai_available or not openai.api_key:
    st.warning(
        "⚠️ OpenAI (for AI narrative) is not fully configured. "
        "Install `openai` in requirements.txt and set `OPENAI_API_KEY` in Streamlit secrets."
    )

# ---------------- HERO / HEADER -------------------
with st.container():
    col_h1, col_h2 = st.columns([2.5, 1.5])

    with col_h1:
        st.markdown(
            """
            <div class="hero-card" title="Hover: This assistant unifies food security, nutrition and youth data into one interactive, AI-ready dashboard.">
                <div class="hero-pill">
                    <span class="dot"></span>
                    AI DATA PIPELINE • FOOD, NUTRITION & YOUTH
                </div>
                <div class="hero-title">
                    AI Food Security, Nutrition & Youth Development Analysis Assistant
                </div>
                <div class="hero-subtitle">
                    Ingest survey data or Kobo submissions, clean and harmonize indicators, and explore 
                    interactive fall-coloured dashboards for food security, dietary diversity and youth agency — 
                    with optional AI narrative reporting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_h2:
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:0.65rem;margin-top:0.2rem;">
                <div class="metric-card" title="Synthetic sample data loads automatically so you can explore the dashboard before uploading real data.">
                    <div class="metric-label">Pipeline status</div>
                    <div class="metric-value">Live & Ready</div>
                    <div class="metric-caption">Synthetic demo data starts automatically.</div>
                </div>
                <div class="metric-card" title="KoboToolbox, CSV, Excel, JSON, TSV, Stata and SPSS are supported for ingest.">
                    <div class="metric-label">Data sources</div>
                    <div class="metric-value">Multi-format</div>
                    <div class="metric-caption">Upload files or pull directly from KoboToolbox.</div>
                </div>
                <div class="metric-card" title="Hover over charts to see precise values, percentages and categories.">
                    <div class="metric-label">Visuals</div>
                    <div class="metric-value">Interactive</div>
                    <div class="metric-caption">Hover for rich tooltips & breakdowns.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================================================
# SCORING HELPERS: HFIAS / HDDS / WDDS
# ==================================================
def categorize_hfias(score):
    if pd.isna(score):
        return np.nan
    if score <= 1:
        return "Food secure"
    elif 2 <= score <= 8:
        return "Mildly food insecure"
    elif 9 <= score <= 16:
        return "Moderately food insecure"
    else:
        return "Severely food insecure"


def compute_hfias_scores(
    df: pd.DataFrame,
    question_cols=None,
    score_col: str = "hfias_score",
    category_col: str = "hfias_category",
):
    if question_cols is None:
        question_cols = [c for c in df.columns if c.lower().startswith("hfias_q")]
    if not question_cols:
        return df
    df[score_col] = df[question_cols].sum(axis=1)
    df[category_col] = df[score_col].apply(categorize_hfias)
    return df


def compute_dietary_diversity_score(
    df: pd.DataFrame,
    food_group_cols,
    score_col: str = "dds_score",
    max_score: int | None = None,
):
    if not food_group_cols:
        return df
    binary = df[food_group_cols].gt(0).astype(int)
    df[score_col] = binary.sum(axis=1)
    if max_score is not None:
        df[score_col] = df[score_col].clip(upper=max_score)
    return df

# ==================================================
# DUMMY DATASET GENERATORS
# ==================================================
def create_dummy_hfias(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sex_head": np.random.choice(["Male", "Female"], n),
        **{f"hfias_q{i}": np.random.randint(0, 4, n) for i in range(1, 10)},
    })
    df = compute_hfias_scores(
        df,
        question_cols=[f"hfias_q{i}" for i in range(1, 10)],
        score_col="hfias_score",
        category_col="hfias_category",
    )
    return df


def create_dummy_wdds(n=500):
    np.random.seed(43)
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": np.random.randint(15, 49, n),
        "sex": "Female",
        "region": np.random.choice(["Urban", "Rural"], n),
        "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n),
    })
    for i in range(1, 10):
        df[f"wdds_fg{i}"] = np.random.binomial(1, 0.6, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"wdds_fg{i}" for i in range(1, 10)], score_col="wdds", max_score=9
    )
    return df


def create_dummy_child_malnutrition(n=500):
    np.random.seed(44)
    df = pd.DataFrame({
        "child_id": range(1, n + 1),
        "age_months": np.random.randint(6, 60, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "weight_kg": np.round(np.random.normal(12, 2.5, n), 1),
        "height_cm": np.round(np.random.normal(90, 8, n), 1),
        "wfh_zscore": np.round(np.random.normal(-0.5, 1.2, n), 2),
    })
    df["malnutrition_type"] = pd.cut(
        df["wfh_zscore"],
        bins=[-10, -3, -2, 100],
        labels=["Severe wasting", "Moderate wasting", "Normal"],
    )
    return df


def create_dummy_consumption_production(n=500):
    np.random.seed(45)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "monthly_food_expense": np.round(np.random.normal(120, 40, n), 2),
        "monthly_income": np.round(np.random.normal(300, 100, n), 2),
        "produces_own_food": np.random.choice([0, 1], n),
        "livestock_count": np.random.poisson(3, n),
    })
    for i in range(1, 13):
        df[f"hdds_fg{i}"] = np.random.binomial(1, 0.7, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"hdds_fg{i}" for i in range(1, 13)], score_col="hdds", max_score=12
    )
    return df


def create_dummy_youth_decision(n=500):
    np.random.seed(46)
    df = pd.DataFrame({
        "youth_id": range(1, n + 1),
        "age": np.random.randint(15, 29, n),
        "sex": np.random.choice(["Male", "Female"], n),
        "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n),
        "employment_status": np.random.choice(
            ["Unemployed", "Employed", "Self-employed", "Student"], n
        ),
        "decision_power_score": np.random.randint(1, 6, n),
        "agency_score": np.random.randint(1, 6, n),
        "hope_future_score": np.random.randint(1, 6, n),
        "financial_literacy_score": np.random.randint(1, 6, n),
        "empathy_score": np.random.randint(1, 6, n),
        "participation_score": np.random.randint(1, 6, n),
    })
    df["received_training"] = np.random.choice([0, 1], n)
    return df


def create_dummy_integrated(n=500):
    np.random.seed(47)
    df = pd.DataFrame({
        "hhid": range(1, n + 1),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sex_head": np.random.choice(["Male", "Female"], n),
        "monthly_income": np.round(np.random.normal(320, 120, n), 2),
        "monthly_food_expense": np.round(np.random.normal(130, 45, n), 2),
        "youth_in_household": np.random.randint(0, 4, n),
        "youth_decision_score": np.random.randint(1, 6, n),
        "youth_agency_score": np.random.randint(1, 6, n),
    })
    for i in range(1, 10):
        df[f"hfias_q{i}"] = np.random.randint(0, 4, n)
    df = compute_hfias_scores(
        df,
        question_cols=[f"hfias_q{i}" for i in range(1, 10)],
        score_col="hfias_score",
        category_col="hfias_category",
    )
    for i in range(1, 13):
        df[f"hdds_fg{i}"] = np.random.binomial(1, 0.7, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"hdds_fg{i}" for i in range(1, 13)], score_col="hdds", max_score=12
    )
    for i in range(1, 10):
        df[f"wdds_fg{i}"] = np.random.binomial(1, 0.6, n)
    df = compute_dietary_diversity_score(
        df, food_group_cols=[f"wdds_fg{i}" for i in range(1, 10)], score_col="wdds", max_score=9
    )
    return df

# ==================================================
# PROGRAM / POLICY-LEVEL NARRATIVE HELPER
# ==================================================
def build_program_narrative(df, dataset_label):
    """
    Build a program/policy-level narrative based on HFIAS, HDDS/WDDS and youth indicators.
    Returns plain text you can display on screen and embed into the PDF.
    """
    name = dataset_label or "Current dataset"
    total_n = len(df)

    lines = []
    lines.append(
        f"Interpretation for this dataset (program/policy level)\n"
        f"Dataset: {name}, Records: {total_n:,}\n"
    )

    # ---------- HFIAS ----------
    if "hfias_score" in df.columns:
        hf_mean = df["hfias_score"].mean()
        hf_mean = float(hf_mean) if pd.notna(hf_mean) else 0.0

        if "hfias_category" in df.columns:
            cat_counts = df["hfias_category"].value_counts(dropna=False)
            cat_pct = (cat_counts / cat_counts.sum() * 100).round(1)
            severe_pct = float(cat_pct.get("Severely food insecure", 0.0))
            mod_pct = float(cat_pct.get("Moderately food insecure", 0.0))
            mild_pct = float(cat_pct.get("Mildly food insecure", 0.0))
            secure_pct = float(cat_pct.get("Food secure", 0.0))
        else:
            severe_pct = mod_pct = mild_pct = secure_pct = 0.0

        lines.append(
            f"- Household food insecurity (HFIAS) shows an average score of {hf_mean:.1f}. "
            f"Approximately {severe_pct:.1f}% of households are severely food insecure, "
            f"and {mod_pct:.1f}% are moderately food insecure, compared to "
            f"{secure_pct:.1f}% who are classified as food secure. This signals "
            f"substantial vulnerability that requires targeted safety nets and livelihood support."
        )

    # ---------- HDDS ----------
    if "hdds" in df.columns:
        hdds_mean = float(df["hdds"].mean())
        if "region" in df.columns:
            reg_hdds = df.groupby("region")["hdds"].mean().sort_values(ascending=False)
            if len(reg_hdds) > 0:
                top_region = str(reg_hdds.index[0])
                top_val = float(reg_hdds.iloc[0])
                lines.append(
                    f"- Household dietary diversity (HDDS) averages {hdds_mean:.1f} food groups in the last 24 hours. "
                    f"Regions such as {top_region} reach up to {top_val:.1f} groups, while others lag behind, "
                    f"highlighting geographic inequities in access to diverse diets."
                )
        else:
            lines.append(
                f"- Household dietary diversity (HDDS) averages {hdds_mean:.1f} food groups in the last 24 hours. "
                f"Scores below 5–6 food groups suggest limited access to nutrient-dense foods among many households."
            )

    # ---------- WDDS ----------
    if "wdds" in df.columns:
        wdds_mean = float(df["wdds"].mean())
        wdds_hi = (df["wdds"] >= 5).mean() * 100
        lines.append(
            f"- Women’s dietary diversity (WDDS) averages {wdds_mean:.1f} food groups. "
            f"Only about {wdds_hi:.1f}% of women achieve a WDDS ≥ 5 food groups, indicating gaps in access to "
            f"micronutrient-dense foods and a need for nutrition-sensitive agriculture, markets, and SBCC."
        )

    # ---------- Youth ----------
    youth_cols = [
        "decision_power_score",
        "agency_score",
        "hope_future_score",
        "financial_literacy_score",
        "empathy_score",
        "participation_score",
    ]
    present_youth_cols = [c for c in youth_cols if c in df.columns]

    if present_youth_cols:
        means = df[present_youth_cols].mean().round(2).to_dict()
        decision = means.get("decision_power_score")
        agency = means.get("agency_score")
        hope = means.get("hope_future_score")
        fin = means.get("financial_literacy_score")

        summary_bits = []
        if decision is not None and not np.isnan(decision):
            summary_bits.append(f"decision-making power ≈ {float(decision):.1f}/5")
        if agency is not None and not np.isnan(agency):
            summary_bits.append(f"agency ≈ {float(agency):.1f}/5")
        if hope is not None and not np.isnan(hope):
            summary_bits.append(f"hope for the future ≈ {float(hope):.1f}/5")
        if fin is not None and not np.isnan(fin):
            summary_bits.append(f"financial literacy ≈ {float(fin):.1f}/5")

        if summary_bits:
            bits_str = ", ".join(summary_bits)
            lines.append(
                f"- Among youth, core empowerment indicators such as {bits_str} suggest mixed but improvable levels "
                f"of agency and economic readiness. Programs that blend life-skills coaching, mentorship, and practical "
                f"financial education could yield sizeable gains in youth pathways to decent work."
            )

        if "received_training" in df.columns:
            trained_pct = (df["received_training"] == 1).mean() * 100
            lines.append(
                f"- Only {trained_pct:.1f}% of youth are flagged as having received structured training. "
                f"This indicates strong scope for scaling cohort-based youth programs that intentionally target "
                f"those with lower decision power and financial literacy scores."
            )

    # ---------- Responsible AI framing ----------
    lines.append(
        "- These insights are derived from aggregated household and youth data; the assistant does not profile "
        "or score individuals. It is designed as a responsible AI decision-support tool to inform program and "
        "policy adjustments (e.g., targeting, coverage, and service mix), not individual-level eligibility decisions."
    )

    return "\n\n".join(lines)

# ==================================================
# SIDEBAR: DATA SOURCE
# ==================================================
st.sidebar.header("🔌 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use sample (dummy) dataset", "Upload your own dataset", "Load from KoboToolbox"],
    help="Synthetic sample data is loaded automatically so you can explore the dashboard immediately.",
)

df = None
dataset_label = None

if data_source == "Use sample (dummy) dataset":
    sample_choice = st.sidebar.selectbox(
        "Select sample dataset",
        [
            "HFIAS household food insecurity",
            "Women’s dietary diversity (WDDS)",
            "Child malnutrition / anthropometry",
            "Food consumption & production",
            "Youth development & decision-making",
            "Integrated multi-topic dataset",
        ],
    )
    if sample_choice == "HFIAS household food insecurity":
        df = create_dummy_hfias()
        dataset_label = "Dummy HFIAS dataset"
    elif sample_choice == "Women’s dietary diversity (WDDS)":
        df = create_dummy_wdds()
        dataset_label = "Dummy WDDS dataset"
    elif sample_choice == "Child malnutrition / anthropometry":
        df = create_dummy_child_malnutrition()
        dataset_label = "Dummy child malnutrition dataset"
    elif sample_choice == "Food consumption & production":
        df = create_dummy_consumption_production()
        dataset_label = "Dummy consumption & production dataset"
    elif sample_choice == "Youth development & decision-making":
        df = create_dummy_youth_decision()
        dataset_label = "Dummy youth decision-making dataset"
    elif sample_choice == "Integrated multi-topic dataset":
        df = create_dummy_integrated()
        dataset_label = "Dummy integrated dataset"

elif data_source == "Upload your own dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload data file",
        type=[
            "csv", "xlsx", "xls", "ods",
            "json", "tsv", "txt",
            "dta", "sav",
            "parquet", "feather",
            "pkl", "pickle",
            "rds", "rda", "rdata",
            "pdf",
        ],
        help="Supports CSV, Excel, TSV/TXT, Stata, SPSS, Parquet, Feather, R data, Python pickles and PDF text.",
    )
    if uploaded_file is not None:
        try:
            name = uploaded_file.name.lower()

            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif name.endswith((".xlsx", ".xls", ".ods")):
                # For .ods you may need 'odfpy' installed
                df = pd.read_excel(uploaded_file)

            elif name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            elif name.endswith((".tsv", ".txt")):
                df = pd.read_csv(uploaded_file, sep="\t")

            elif name.endswith(".dta"):
                # Stata
                df = pd.read_stata(uploaded_file)

            elif name.endswith(".sav"):
                # SPSS via pyreadstat
                try:
                    import pyreadstat
                    df, meta = pyreadstat.read_sav(uploaded_file)
                except ImportError:
                    st.error(
                        "Reading SPSS (.sav) requires the 'pyreadstat' package. "
                        "Add 'pyreadstat' to requirements.txt."
                    )
                    df = None

            elif name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)

            elif name.endswith(".feather"):
                df = pd.read_feather(uploaded_file)

            elif name.endswith((".pkl", ".pickle")):
                import pickle
                obj = pickle.load(uploaded_file)
                if isinstance(obj, pd.DataFrame):
                    df = obj
                else:
                    st.error(
                        "The Python pickle file does not contain a pandas DataFrame. "
                        "Please pickle a DataFrame object."
                    )
                    df = None

            elif name.endswith((".rds", ".rda", ".rdata")):
                # R data via pyreadr
                try:
                    import pyreadr
                    result = pyreadr.read_r(uploaded_file)
                    if len(result) == 0:
                        st.error("No objects found in the R data file.")
                        df = None
                    else:
                        # Take the first object that is a DataFrame
                        df = None
                        for obj_name, obj_val in result.items():
                            if isinstance(obj_val, pd.DataFrame):
                                df = obj_val
                                dataset_label = f"R data ({obj_name})"
                                break
                        if df is None:
                            st.error(
                                "The R data file does not contain a pandas-compatible DataFrame."
                            )
                except ImportError:
                    st.error(
                        "Reading R data (.rds/.rda/.RData) requires the 'pyreadr' package. "
                        "Add 'pyreadr' to requirements.txt."
                    )
                    df = None

            elif name.endswith(".pdf"):
                # PDF support: convert pages to text using pdfplumber
                try:
                    import pdfplumber

                    pages = []
                    with pdfplumber.open(uploaded_file) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            pages.append({"page": i + 1, "text": text})
                    df = pd.DataFrame(pages)
                    dataset_label = "PDF text (page-level) dataset"
                    st.info(
                        "📄 PDF loaded as page-level text. You can generate AI narrative "
                        "from this text, but numeric analysis may be limited."
                    )
                except Exception as e:
                    st.error(
                        f"Error reading PDF with pdfplumber: {e}. "
                        "Check that 'pdfplumber' is in your requirements.txt."
                    )
                    df = None

            else:
                st.error("Unsupported file format.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

elif data_source == "Load from KoboToolbox":
    st.sidebar.markdown("### 🌐 KoboToolbox Connection")
    kobo_server = st.sidebar.text_input(
        "Kobo server URL",
        value="https://eu.kobotoolbox.org",
    )
    default_kobo_token = st.secrets.get("KOBO_TOKEN", "")
    kobo_token = st.sidebar.text_input(
        "Kobo API Token",
        type="password",
        value=default_kobo_token,
    )
    kobo_asset_uid = st.sidebar.text_input(
        "Form / Asset UID",
        value="aRcHyjoYEn6fCkHuEX4QYm",
    )
    load_kobo = st.sidebar.button("Load data from Kobo")
    if load_kobo:
        if not (kobo_server and kobo_token and kobo_asset_uid):
            st.error("Please provide server URL, API token, and asset UID.")
        else:
            try:
                kobo_server = kobo_server.rstrip("/")
                url = f"{kobo_server}/api/v2/assets/{kobo_asset_uid}/data/?format=json"
                headers = {"Authorization": f"Token {kobo_token}"}
                st.info("Requesting data from KoboToolbox...")
                resp = requests.get(url, headers=headers)
                if resp.status_code != 200:
                    st.error(
                        f"Error fetching data from Kobo (status {resp.status_code}): "
                        f"{resp.text[:400]}"
                    )
                else:
                    data_json = resp.json()
                    records = data_json.get("results", data_json)
                    if not records:
                        st.warning("No submissions found for this asset.")
                    else:
                        df = pd.DataFrame.from_records(records)
                        dataset_label = f"Kobo data (asset {kobo_asset_uid})"
                        st.success("Data loaded successfully from KoboToolbox.")
                        system_cols = [c for c in df.columns if c.startswith("_")]
                        if system_cols:
                            st.info(
                                f"Dropping Kobo system columns from analysis: {system_cols}"
                            )
                            df = df.drop(columns=system_cols)
            except Exception as e:
                st.error(f"Error loading data from KoboToolbox: {e}")

if df is None:
    st.info(
        "Select a sample dataset, upload a file, or load from KoboToolbox to begin. "
        "Sample HFIAS data is selected by default on first load."
    )
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

tab_overview, tab_auto, tab_desc, tab_visual, tab_ai = st.tabs(
    ["📂 Overview", "📈 Auto Insights", "📊 Descriptives & Crosstabs", "📈 Visual Explorer", "🧠 AI Narrative & Downloads"]
)

narrative_chunks = []

# ==================================================
# OVERVIEW TAB
# ==================================================
with tab_overview:
    st.markdown(
        '<div class="section-title">DATA SNAPSHOT</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.4, 1.4, 1.2])
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-caption">Survey records / respondents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{df.shape[1]:,}</div>
                <div class="metric-caption">Indicators & attributes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Variable mix</div>
                <div class="metric-value">{len(numeric_cols)} / {len(categorical_cols)}</div>
                <div class="metric-caption">Numeric / categorical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"**Dataset loaded:** `{dataset_label or 'Uploaded dataset'}` · "
        f"Shape: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**"
    )
    with st.expander("🔍 Preview first rows & data types", expanded=True):
        st.markdown('<div class="dataframe-card">', unsafe_allow_html=True)
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("**Numeric columns**", numeric_cols)
        st.write("**Categorical/text columns**", categorical_cols)

# ==================================================
# SIDEBAR ANALYSIS OPTIONS
# ==================================================
st.sidebar.header("⚙️ Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose analysis mode",
    ["AI auto-detect analysis", "Use custom analysis script"],
)

st.sidebar.header("📊 Descriptive & Crosstab Options")
selected_numeric_by_group = st.sidebar.multiselect(
    "Numeric variables for descriptive by categories (mean, std, etc.)",
    options=numeric_cols,
)
selected_group_vars = st.sidebar.multiselect(
    "Grouping categorical variables (e.g., region, sex, education)",
    options=categorical_cols,
)
crosstab_var1 = st.sidebar.selectbox(
    "Crosstab variable 1 (categorical)",
    ["(none)"] + categorical_cols if categorical_cols else ["(none)"],
)
crosstab_var2 = st.sidebar.selectbox(
    "Crosstab variable 2 (categorical)",
    ["(none)"] + categorical_cols if categorical_cols else ["(none)"],
)
selected_cats_for_freq = st.sidebar.multiselect(
    "Categorical variables for frequency plots (bar/pie)",
    options=categorical_cols,
)

# ==================================================
# AUTO / CUSTOM ANALYSIS TAB
# ==================================================
with tab_auto:
    if analysis_mode == "Use custom analysis script":
        st.markdown(
            '<div class="section-title">CUSTOM SCRIPT–DRIVEN ANALYSIS</div>',
            unsafe_allow_html=True,
        )
        script_type = st.selectbox(
            "Type of script you want to reference",
            ["Stata (.do)", "SPSS (.sps)", "Python code (text)"],
        )
        script_text = st.text_area(
            "Paste your analysis script here (for now, used as guidance for AI + summaries)",
            height=200,
        )
        st.markdown("#### Basic descriptive analysis based on your dataset")
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            st.dataframe(desc)
            narrative_chunks.append(
                "Descriptive statistics (custom script mode):\n" + desc.to_string()
            )
        else:
            st.warning("No numeric columns detected for descriptive statistics.")
    else:
        st.markdown(
            '<div class="section-title">AUTO-DETECTED FOOD SECURITY & YOUTH ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        # HFIAS
        if "hfias_score" in df.columns:
            st.markdown("##### HFIAS severity & distribution")
            hfias = df["hfias_score"]
            st.dataframe(hfias.describe().to_frame().T)
            cat_col = "hfias_category" if "hfias_category" in df.columns else None
            if not cat_col:
                df = compute_hfias_scores(df, None, "hfias_score", "hfias_category")
                cat_col = "hfias_category"
            total_n = len(df)
            hfias_summary = (
                df.groupby(cat_col)["hfias_score"]
                .agg(count="size", mean="mean")
                .reset_index()
            )
            hfias_summary["percent"] = (hfias_summary["count"] / total_n * 100).round(1)
            hfias_summary["mean"] = hfias_summary["mean"].round(2)
            st.dataframe(hfias_summary)
            fig_hfias = px.bar(
                hfias_summary,
                x=cat_col,
                y="count",
                color=cat_col,
                title="HFIAS severity distribution (hover for % & mean)",
                hover_data={"percent": True, "mean": True},
            )
            st.plotly_chart(fig_hfias, use_container_width=True)
            narrative_chunks.append(
                "HFIAS distribution (count, mean score, and % of total by category):\n"
                + hfias_summary.to_string(index=False)
            )

        # HDDS / WDDS
        for col in ["hdds", "wdds"]:
            if col in df.columns:
                st.markdown(f"##### {col.upper()} distribution")
                st.write(df[col].describe())
                fig_dds = px.histogram(
                    df,
                    x=col,
                    nbins=10,
                    title=f"Distribution of {col.upper()} (hover for counts)",
                )
                st.plotly_chart(fig_dds, use_container_width=True)
                narrative_chunks.append(
                    f"{col.upper()} distribution summary:\n"
                    + df[col].describe().to_string()
                )

        # Child anthropometry
        if "wfh_zscore" in df.columns:
            st.markdown("##### Child weight-for-height z-scores & malnutrition")
            st.write(df["wfh_zscore"].describe())
            fig_z = px.histogram(
                df,
                x="wfh_zscore",
                nbins=12,
                title="Distribution of WFH Z-scores",
            )
            st.plotly_chart(fig_z, use_container_width=True)
            narrative_chunks.append(
                "WFH z-score distribution:\n"
                + df["wfh_zscore"].describe().to_string()
            )
            if "malnutrition_type" in df.columns:
                mal_counts = df["malnutrition_type"].value_counts(dropna=False)
                mal_tbl = pd.DataFrame({
                    "malnutrition_type": mal_counts.index.astype(str),
                    "count": mal_counts.values,
                })
                mal_tbl["percent"] = (
                    mal_tbl["count"] / mal_tbl["count"].sum() * 100
                ).round(1)
                st.dataframe(mal_tbl)
                fig_mal = px.bar(
                    mal_tbl,
                    x="malnutrition_type",
                    y="count",
                    color="malnutrition_type",
                    title="Malnutrition type (hover for %)",
                    hover_data={"percent": True},
                )
                st.plotly_chart(fig_mal, use_container_width=True)
                narrative_chunks.append(
                    "Malnutrition type frequencies (count and %):\n"
                    + mal_tbl.to_string(index=False)
                )

        # Youth
        st.markdown("##### Youth development & empowerment indicators")
        youth_num_keywords = [
            "decision", "power", "agency", "aspiration", "hope", "future",
            "financial", "finance", "literacy", "saving", "savings",
            "budget", "empathy", "empath", "participation", "voice", "leadership"
        ]
        youth_cat_keywords = [
            "youth", "training", "program", "cohort", "group",
            "employment", "employed", "self_emp", "business",
            "mentor", "club", "volunteer"
        ]
        youth_numeric_cols = [c for c in numeric_cols if any(k in c.lower() for k in youth_num_keywords)]
        youth_categorical_cols = [c for c in categorical_cols if any(k in c.lower() for k in youth_num_keywords + youth_cat_keywords)]
        if youth_numeric_cols or youth_categorical_cols:
            st.success(
                f"Detected youth-relevant variables: "
                f"{len(youth_numeric_cols)} numeric, {len(youth_categorical_cols)} categorical."
            )
            if youth_numeric_cols:
                st.markdown("###### Youth numeric indicators")
                youth_desc = df[youth_numeric_cols].describe().T
                st.dataframe(youth_desc)
                narrative_chunks.append(
                    "Youth numeric indicators descriptives:\n"
                    + youth_desc.to_string()
                )
                for col in youth_numeric_cols[:8]:
                    fig_y = px.histogram(
                        df,
                        x=col,
                        nbins=10,
                        title=f"Distribution of {col}",
                    )
                    st.plotly_chart(fig_y, use_container_width=True)
            if youth_numeric_cols and "sex" in df.columns:
                st.markdown("###### Youth indicators by sex")
                for col in youth_numeric_cols:
                    grouped = df.groupby("sex")[col].mean().reset_index()
                    st.dataframe(grouped)
                    fig_sex = px.bar(
                        grouped,
                        x="sex",
                        y=col,
                        title=f"Mean {col} by sex",
                    )
                    st.plotly_chart(fig_sex, use_container_width=True)
                    narrative_chunks.append(
                        f"Mean {col} by sex:\n{grouped.to_string(index=False)}"
                    )
            if youth_numeric_cols and "region" in df.columns:
                st.markdown("###### Youth indicators by region")
                for col in youth_numeric_cols:
                    grouped_r = df.groupby("region")[col].mean().reset_index()
                    st.dataframe(grouped_r)
                    fig_reg = px.bar(
                        grouped_r,
                        x="region",
                        y=col,
                        title=f"Mean {col} by region",
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
                    narrative_chunks.append(
                        f"Mean {col} by region:\n{grouped_r.to_string(index=False)}"
                    )
            if youth_categorical_cols:
                st.markdown("###### Youth categorical variables")
                total_n = len(df)
                for col in youth_categorical_cols:
                    vc = df[col].value_counts(dropna=False)
                    freq_tbl = pd.DataFrame({
                        col: vc.index.astype(str),
                        "count": vc.values,
                    })
                    freq_tbl["percent"] = (
                        freq_tbl["count"] / total_n * 100
                    ).round(1)
                    st.dataframe(freq_tbl)
                    fig_cat = px.bar(
                        freq_tbl,
                        x=col,
                        y="count",
                        title=f"{col} (bar chart, hover for %)",
                        hover_data={"percent": True},
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                    if vc.shape[0] <= 10:
                        fig_pie = px.pie(
                            freq_tbl,
                            names=col,
                            values="count",
                            title=f"{col} (pie chart)",
                            hover_data={"percent": True},
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    narrative_chunks.append(
                        f"Youth categorical {col} (count and %):\n"
                        + freq_tbl.to_string(index=False)
                    )
        else:
            st.info(
                "No youth-specific variables detected by name. "
                "You can still use the general descriptive and crosstab options."
            )

# ==================================================
# DESCRIPTIVES & CROSSTABS TAB
# ==================================================
with tab_desc:
    st.markdown(
        '<div class="section-title">DESCRIPTIVE ANALYSIS & CROSSTABS</div>',
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([2, 1])

    if selected_numeric_by_group and selected_group_vars:
        with left_col:
            st.markdown("#### Numeric variables by selected categories")
            total_n = len(df)
            group_counts = df.groupby(selected_group_vars).size().rename("group_count")
            group_perc = (group_counts / total_n * 100).rename("group_percent")
            grouped_stats = df.groupby(selected_group_vars)[selected_numeric_by_group].agg(
                ["mean", "std", "count", "min", "max"]
            )
            grouped = pd.concat([grouped_stats, group_counts, group_perc], axis=1)
            new_cols = []
            for col in grouped.columns:
                if isinstance(col, tuple):
                    new_cols.append("_".join([str(c) for c in col if c != ""]).strip("_"))
                else:
                    new_cols.append(str(col))
            grouped.columns = new_cols
            st.dataframe(grouped)
            narrative_chunks.append(
                f"Numeric descriptives by {selected_group_vars} for {selected_numeric_by_group} "
                f"(including group_count and group_percent):\n{grouped.to_string()}"
            )
        primary_group = selected_group_vars[0]
        with right_col:
            st.markdown(f"#### Mean by {primary_group}")
            for num_var in selected_numeric_by_group:
                mean_df = (
                    df.groupby(primary_group)[num_var]
                    .mean()
                    .reset_index()
                    .rename(columns={num_var: "mean_value"})
                )
                fig_mean = px.bar(
                    mean_df,
                    x=primary_group,
                    y="mean_value",
                    title=f"Mean {num_var} by {primary_group}",
                )
                st.plotly_chart(fig_mean, use_container_width=True)

    if selected_cats_for_freq:
        with left_col:
            st.markdown("#### Frequency tables for selected categoricals")
            total_n = len(df)
            for cat in selected_cats_for_freq:
                vc = df[cat].value_counts(dropna=False)
                freq_tbl = pd.DataFrame({
                    cat: vc.index.astype(str),
                    "count": vc.values,
                })
                freq_tbl["percent"] = (
                    freq_tbl["count"] / total_n * 100
                ).round(1)
                st.write(f"**{cat}**")
                st.dataframe(freq_tbl)
                narrative_chunks.append(
                    f"Frequencies for {cat} (count and % of total):\n"
                    + freq_tbl.to_string(index=False)
                )
        with right_col:
            st.markdown("#### Categorical plots")
            for cat in selected_cats_for_freq:
                vc = df[cat].value_counts(dropna=False)
                freq_tbl = pd.DataFrame({
                    cat: vc.index.astype(str),
                    "count": vc.values,
                })
                fig_cat = px.bar(
                    freq_tbl,
                    x=cat,
                    y="count",
                    title=f"{cat} (bar chart)",
                )
                st.plotly_chart(fig_cat, use_container_width=True)
                if vc.shape[0] <= 10:
                    fig_cat_pie = px.pie(
                        freq_tbl,
                        names=cat,
                        values="count",
                        title=f"{cat} (pie chart)",
                    )
                    st.plotly_chart(fig_cat_pie, use_container_width=True)

    if (
        crosstab_var1 != "(none)"
        and crosstab_var2 != "(none)"
        and crosstab_var1 != crosstab_var2
    ):
        st.markdown("#### Crosstab between two categorical variables")
        with left_col:
            xtab = pd.crosstab(df[crosstab_var1], df[crosstab_var2], dropna=False)
            st.dataframe(xtab)
            narrative_chunks.append(
                f"Crosstab counts for {crosstab_var1} x {crosstab_var2}:\n"
                + xtab.to_string()
            )
            xtab_pct = xtab.div(xtab.sum(axis=1), axis=0) * 100
            st.dataframe(xtab_pct.round(1))
        with right_col:
            xtab_pct_reset = xtab_pct.reset_index().melt(
                id_vars=crosstab_var1,
                var_name=crosstab_var2,
                value_name="row_percent",
            )
            fig_xtab = px.bar(
                xtab_pct_reset,
                x=crosstab_var1,
                y="row_percent",
                color=crosstab_var2,
                title=f"{crosstab_var1} × {crosstab_var2} (row % stacked)",
            )
            st.plotly_chart(fig_xtab, use_container_width=True)

    st.markdown("#### General descriptive statistics")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)
        narrative_chunks.append("Overall numeric descriptives:\n" + desc.to_string())
    if categorical_cols:
        st.markdown("#### Key categorical distributions (top 10)")
        total_n = len(df)
        for col in categorical_cols[:10]:
            vc = df[col].value_counts(dropna=False)
            freq_tbl = pd.DataFrame({
                col: vc.index.astype(str),
                "count": vc.values,
            })
            freq_tbl["percent"] = (
                freq_tbl["count"] / total_n * 100
            ).round(1)
            st.dataframe(freq_tbl)
            narrative_chunks.append(
                f"Value counts for {col} (count and %):\n"
                + freq_tbl.to_string(index=False)
            )

# ==================================================
# VISUAL EXPLORER TAB (Histograms, Boxplots, Pie, Scatter)
# ==================================================
with tab_visual:
    st.markdown(
        '<div class="section-title">VISUAL EXPLORER (HISTOGRAM, BOX, PIE, SCATTER)</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.5, 1.5])

    # --- Numeric visual controls ---
    with col_left:
        st.markdown("##### Numeric visuals")
        num_for_hist = st.multiselect(
            "Numeric variables for histograms",
            options=numeric_cols,
            default=numeric_cols[:3] if numeric_cols else [],
        )
        num_for_box = st.multiselect(
            "Numeric variables for boxplots",
            options=numeric_cols,
            default=numeric_cols[:3] if numeric_cols else [],
        )
        box_group = st.selectbox(
            "Optional group-by (for boxplots)",
            options=["(none)"] + categorical_cols if categorical_cols else ["(none)"],
            index=0,
        )

    # --- Categorical visual controls ---
    with col_right:
        st.markdown("##### Categorical visuals")
        cat_for_bar_pie = st.multiselect(
            "Categorical variables for bar + pie charts",
            options=categorical_cols,
            default=categorical_cols[:2] if categorical_cols else [],
        )
        st.markdown("##### Scatter (numeric × numeric)")
        scatter_x = st.selectbox("X-axis", options=["(none)"] + numeric_cols)
        scatter_y = st.selectbox(
            "Y-axis",
            options=["(none)"] + numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
        )

    st.markdown("---")

    # --- Histograms ---
    if num_for_hist:
        st.markdown("###### Histograms")
        for col in num_for_hist:
            fig_h = px.histogram(
                df,
                x=col,
                nbins=20,
                title=f"Histogram of {col}",
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # --- Boxplots ---
    if num_for_box:
        st.markdown("###### Boxplots")
        for col in num_for_box:
            if box_group != "(none)" and box_group in categorical_cols:
                fig_b = px.box(
                    df,
                    x=box_group,
                    y=col,
                    title=f"Boxplot of {col} by {box_group}",
                )
            else:
                fig_b = px.box(
                    df,
                    y=col,
                    title=f"Boxplot of {col}",
                )
            st.plotly_chart(fig_b, use_container_width=True)

    # --- Bar + Pie for categoricals ---
    if cat_for_bar_pie:
        st.markdown("###### Bar & Pie Charts for Categoricals")
        total_n = len(df)
        for cat in cat_for_bar_pie:
            vc = df[cat].value_counts(dropna=False)
            freq_tbl = pd.DataFrame({
                cat: vc.index.astype(str),
                "count": vc.values,
            })
            freq_tbl["percent"] = (freq_tbl["count"] / total_n * 100).round(1)
            fig_bar = px.bar(
                freq_tbl,
                x=cat,
                y="count",
                title=f"{cat} – bar chart (hover for %)",
                hover_data={"percent": True},
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            if vc.shape[0] <= 12:
                fig_pie = px.pie(
                    freq_tbl,
                    names=cat,
                    values="count",
                    title=f"{cat} – pie chart",
                    hover_data={"percent": True},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # --- Scatter ---
    if scatter_x != "(none)" and scatter_y != "(none)" and scatter_x != scatter_y:
        st.markdown("###### Scatterplot")
        fig_s = px.scatter(
            df,
            x=scatter_x,
            y=scatter_y,
            title=f"Scatter: {scatter_x} vs {scatter_y}",
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ==================================================
# AI NARRATIVE & DOWNLOADS
# ==================================================
with tab_ai:
    st.markdown(
        '<div class="section-title">AI NARRATIVE REPORT & EXPORTS</div>',
        unsafe_allow_html=True,
    )

    # ---------- Data-driven program/policy narrative (no OpenAI required) ----------
    st.markdown("##### Program-/Policy-level Interpretation (Data-driven)")

    program_narrative = build_program_narrative(df, dataset_label)
    # Wrap in styled narrative card for readability
    st.markdown(
        f"<div class='narrative-card'>{program_narrative}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### AI Narrative Report (optional, using OpenAI)")

    if (not openai_available) or (not openai.api_key):
        st.warning(
            "OpenAI is not fully available. Install the `openai` package and "
            "set `OPENAI_API_KEY` in Streamlit secrets to enable the AI-generated narrative."
        )
        ai_text = None
    else:
        user_prompt = st.text_area(
            "Optional: refine what you want the AI to focus on",
            value=(
                "Summarize the dataset, describe key findings on food security, "
                "dietary diversity, youth decision-making and agency, hope for the future, "
                "financial literacy, empathy, and income/consumption where applicable. "
                "Highlight any gender or regional differences and potential program implications."
            ),
        )

        ai_text = None
        if st.button("Generate AI Narrative"):
            context_sample = df.head(10).to_dict()
            combined_narrative = (
                "\n\n".join(narrative_chunks) if narrative_chunks else "No prior summaries."
            )
            prompt = (
                "You are an expert data analyst working on food security, nutrition, and youth development.\n"
                "Write a clear, non-technical narrative (1–2 pages) summarizing the key insights "
                "from the analysis below, and suggest 3–5 program or policy implications.\n\n"
                f"USER FOCUS: {user_prompt}\n\n"
                "ANALYSIS SUMMARIES:\n"
                f"{combined_narrative}\n\n"
                "DATA PREVIEW (first 10 rows as dict):\n"
                f"{context_sample}\n"
            )
            try:
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                ai_text = completion.choices[0].message.content
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

    # Show AI narrative (if any) in the same dark card style for readability
    if ai_text:
        st.markdown(
            f"<div class='narrative-card'>{ai_text}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### Downloads")

    # CSV Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download current dataset as CSV",
        data=csv_bytes,
        file_name="dataset_processed.csv",
        mime="text/csv",
    )

    # PDF: narrative + charts
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: narrative text (program-level)
        try:
            fig_text = plt.figure(figsize=(8.27, 11.69))  # A4-ish
            fig_text.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

            title = "Program-level Interpretation – Food Security, Nutrition & Youth"
            fig_text.text(
                0.5, 0.95, title, ha="center", va="top", fontsize=14, weight="bold"
            )

            # Wrap text for the PDF page
            plain = program_narrative
            lines = plain.split("\n")
            y = 0.9
            dy = 0.03
            for line in lines:
                if not line.strip():
                    y -= dy / 2
                    continue
                wrapped = textwrap.wrap(line, width=95)
                for wline in wrapped:
                    fig_text.text(0.03, y, wline, ha="left", va="top", fontsize=9)
                    y -= dy
                    if y < 0.05:
                        break
                if y < 0.05:
                    break

            pdf.savefig(fig_text)
            plt.close(fig_text)
        except Exception:
            # If narrative rendering fails, just skip the text page and continue with charts
            pass

        # Additional pages: numeric histograms
        for col in numeric_cols[:12]:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=12)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            pdf.savefig(fig)
            plt.close(fig)

    pdf_buffer.seek(0)

    st.download_button(
        "⬇️ Download PDF report (narrative + charts)",
        data=pdf_buffer,
        file_name="analysis_report.pdf",
        mime="application/pdf",
    )
