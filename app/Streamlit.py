"""
streamlit_app.py
================
Telecom Customer Churn — Interactive Prediction App

Run:
    streamlit run app/streamlit_app.py

Requirements:
    pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap joblib matplotlib seaborn plotly
"""

import os, sys, warnings, io
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, f1_score, average_precision_score,
)

# ── allow running from project root ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
BINARY_MAP = {
    "Yes": 1, "No": 0, "Male": 1, "Female": 0,
    "No internet service": 0, "No phone service": 0,
}
BINARY_COLS = [
    "gender","Partner","Dependents","PhoneService","PaperlessBilling",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies",
]
OHE_COLS = ["MultipleLines","InternetService","Contract","PaymentMethod","TenureGroup"]
NUM_COLS = ["tenure","MonthlyCharges","TotalCharges","AvgMonthlySpend","ServicesCount"]
SERVICE_COLS = [
    "PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies",
]
TENURE_BINS   = [0,12,24,48,60,72]
TENURE_LABELS = ["0-12","13-24","25-48","49-60","61-72"]
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

# ─────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard · Telecom",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.3rem;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem !important;
    font-weight: 700;
    color: #58a6ff;
}
[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.82rem; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #0f1923 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(88,166,255,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: #58a6ff; }
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin: 0;
    max-width: 560px;
}

/* ── section heading ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.7rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 50px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.risk-high   { background:#2d1117; color:#f85149; border:1px solid #f85149; }
.risk-medium { background:#271d0e; color:#e3b341; border:1px solid #e3b341; }
.risk-low    { background:#0f2619; color:#3fb950; border:1px solid #3fb950; }

/* ── prob bar ── */
.prob-track {
    background: #21262d;
    border-radius: 6px;
    height: 14px;
    margin: 0.6rem 0;
    overflow: hidden;
}
.prob-fill-high   { height:100%; background:linear-gradient(90deg,#da3633,#f85149); border-radius:6px; transition:width 0.4s; }
.prob-fill-medium { height:100%; background:linear-gradient(90deg,#b08800,#e3b341); border-radius:6px; transition:width 0.4s; }
.prob-fill-low    { height:100%; background:linear-gradient(90deg,#238636,#3fb950); border-radius:6px; transition:width 0.4s; }

/* ── result card ── */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-top: 0.8rem;
}

/* ── info box ── */
.info-box {
    background: #0f1923;
    border: 1px solid #1f6feb55;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #8b949e;
    margin-top: 1rem;
}

/* ── table ── */
.dataframe { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px 8px 0 0;
    color: #8b949e;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #58a6ff !important;
    border-color: #58a6ff !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.55rem 1.8rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

/* ── inputs ── */
.stSelectbox [data-baseweb="select"] > div,
.stSlider, .stNumberInput input {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── plotly ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    model         = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


@st.cache_data(show_spinner=False)
def load_test_data():
    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()
    return X_test, y_test


def preprocess_single(raw: dict, scaler, feature_names):
    df = pd.DataFrame([raw])
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    if df["TotalCharges"].iloc[0] == 0:
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    service_present = [c for c in SERVICE_COLS if c in df.columns]
    df["ServicesCount"] = df[service_present].apply(lambda r: (r == "Yes").sum(), axis=1)
    t = df["tenure"].iloc[0]
    df["AvgMonthlySpend"] = df["TotalCharges"] / t if t > 0 else df["MonthlyCharges"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP).fillna(df[col])
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=False)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    num_present = [c for c in NUM_COLS if c in df.columns]
    df[num_present] = scaler.transform(df[num_present])
    return df


def risk_class(prob):
    if prob >= 0.70: return "high",   "🔴 HIGH RISK",   "risk-high",   "prob-fill-high"
    if prob >= 0.40: return "medium", "🟡 MEDIUM RISK", "risk-medium", "prob-fill-medium"
    return               "low",    "🟢 LOW RISK",    "risk-low",    "prob-fill-low"


def dark_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    return fig, ax


def dark_fig_multi(nrows=1, ncols=1, figsize=(10, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in ax_list:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
    return fig, axes


# ─────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────
try:
    model, scaler, feature_names = load_artifacts()
    X_test, y_test = load_test_data()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_error = str(e)

# ─────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:0.6rem 0 1.2rem 0;'>
      <span style='font-size:1.6rem;'>📡</span>
      <div>
        <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:1.05rem;color:#e6edf3;'>ChurnGuard</div>
        <div style='font-size:0.72rem;color:#58a6ff;'>Telecom · ML Dashboard</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Navigation")
    page = st.radio(
        label="",
        options=["🏠 Overview", "🔮 Predict Single", "📂 Batch Predict", "📊 Model Performance", "🔍 SHAP Explainability"],
        label_visibility="collapsed",
    )

    st.markdown("## Model Info")
    if artifacts_ok:
        st.markdown(f"""
        <div style='font-size:0.82rem;color:#8b949e;line-height:1.8;'>
        <b style='color:#58a6ff;'>Algorithm</b><br>{type(model).__name__}<br><br>
        <b style='color:#58a6ff;'>Features</b><br>{len(feature_names)}<br><br>
        <b style='color:#58a6ff;'>Test samples</b><br>{len(X_test):,}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Artifacts not found")

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#484f58;'>Built with Streamlit · XGBoost · SHAP</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Guard — no artifacts
# ─────────────────────────────────────────────────────────────────
if not artifacts_ok:
    st.error(f"⚠️ Could not load model artifacts from `{ARTIFACTS_DIR}/`")
    st.info("Run notebooks 02 and 03 first (or `python src/preprocess.py && python src/train.py`) to generate the artifacts.")
    st.code(artifact_error)
    st.stop()


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
      <p class='hero-title'>Telecom <span>Churn Intelligence</span></p>
      <p class='hero-sub'>Predict customer churn before it happens — powered by XGBoost & SHAP explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    y_proba_all = model.predict_proba(X_test)[:, 1]
    y_pred_all  = (y_proba_all >= 0.5).astype(int)
    auc   = roc_auc_score(y_test, y_proba_all)
    f1    = f1_score(y_test, y_pred_all)
    ap    = average_precision_score(y_test, y_proba_all)
    churn_rate = y_test.mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",    f"{auc:.3f}",        "Test set")
    c2.metric("F1-Score",   f"{f1:.3f}",          "Churn class")
    c3.metric("Avg Prec.",  f"{ap:.3f}",          "Precision-Recall")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%", "Test set")

    st.markdown("<div class='section-title'>What This App Does</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='result-card'>
          <div style='font-size:1.6rem;margin-bottom:0.5rem;'>🔮</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>Single Prediction</div>
          <div style='font-size:0.86rem;color:#8b949e;'>Input one customer's details and get an instant churn probability with risk rating.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='result-card'>
          <div style='font-size:1.6rem;margin-bottom:0.5rem;'>📂</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>Batch Scoring</div>
          <div style='font-size:0.86rem;color:#8b949e;'>Upload a CSV of customers and download predictions with risk labels in seconds.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='result-card'>
          <div style='font-size:1.6rem;margin-bottom:0.5rem;'>🔍</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>SHAP Insights</div>
          <div style='font-size:0.86rem;color:#8b949e;'>Understand which features drive churn globally and for individual customers.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Key Business Insights</div>", unsafe_allow_html=True)
    insights = [
        ("📋 Contract Type",     "Month-to-month customers churn at ~42% vs 3% on 2-year plans."),
        ("⏱️ Tenure",            "Customers in their first 12 months are at highest risk — prioritize early engagement."),
        ("🌐 Internet Service",  "Fiber optic users churn more — often due to pricing or service quality issues."),
        ("🔒 Online Security",   "Customers without security/tech support add-ons churn significantly more."),
        ("💳 Payment Method",    "Electronic check payers have the highest churn; auto-pay reduces risk."),
        ("👴 Senior Citizens",   "Seniors churn at ~41% vs 24% for non-seniors — dedicated plans can help."),
    ]
    for i in range(0, len(insights), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(insights):
                icon_title, body = insights[i + j]
                col.markdown(f"""
                <div class='result-card' style='margin-bottom:0.6rem;'>
                  <div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:#58a6ff;margin-bottom:0.3rem;'>{icon_title}</div>
                  <div style='font-size:0.86rem;color:#8b949e;'>{body}</div>
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Single
# ═════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Single":
    st.markdown("<p class='hero-title' style='margin-bottom:0.2rem;'>🔮 Single Customer <span style='color:#58a6ff;'>Predictor</span></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.6rem;'>Fill in the customer profile and click Predict Churn.</p>", unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("<div class='section-title'>👤 Demographics</div>", unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        gender        = dc1.selectbox("Gender",          ["Male", "Female"])
        senior        = dc2.selectbox("Senior Citizen",  ["No", "Yes"])
        partner       = dc3.selectbox("Partner",         ["Yes", "No"])
        dependents    = dc4.selectbox("Dependents",      ["No", "Yes"])

        st.markdown("<div class='section-title'>📱 Services</div>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        phone_svc     = sc1.selectbox("Phone Service",     ["Yes", "No"])
        multi_lines   = sc2.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
        internet_svc  = sc3.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
        online_sec    = sc4.selectbox("Online Security",   ["No", "Yes", "No internet service"])

        sc5, sc6, sc7, sc8 = st.columns(4)
        online_bkp    = sc5.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot   = sc6.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_sup      = sc7.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        stream_tv     = sc8.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])

        sc9, _, _, _ = st.columns(4)
        stream_mv     = sc9.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        st.markdown("<div class='section-title'>💳 Account & Billing</div>", unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3)
        contract      = ac1.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
        paperless     = ac2.selectbox("Paperless Billing",["Yes", "No"])
        payment       = ac3.selectbox("Payment Method",   [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"
        ])

        st.markdown("<div class='section-title'>💰 Charges & Tenure</div>", unsafe_allow_html=True)
        nc1, nc2, nc3 = st.columns(3)
        tenure         = nc1.slider("Tenure (months)",    0, 72, 12)
        monthly_charge = nc2.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charge   = nc3.number_input("Total Charges ($)",   0.0, 10000.0,
                                           float(tenure * monthly_charge), step=10.0)

        submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        raw = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_svc,
            "MultipleLines": multi_lines, "InternetService": internet_svc,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charge, "TotalCharges": total_charge,
        }

        X_input = preprocess_single(raw, scaler, feature_names)
        prob    = model.predict_proba(X_input)[0, 1]
        level, label, badge_cls, fill_cls = risk_class(prob)

        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f"""
            <div class='result-card' style='text-align:center;'>
              <div style='font-size:3.2rem;font-family:Space Grotesk,sans-serif;font-weight:700;color:{"#f85149" if level=="high" else "#e3b341" if level=="medium" else "#3fb950"};'>
                {prob*100:.1f}%
              </div>
              <div style='color:#8b949e;font-size:0.88rem;margin-bottom:0.8rem;'>Churn Probability</div>
              <span class='risk-badge {badge_cls}'>{label}</span>
              <div class='prob-track' style='margin-top:1rem;'>
                <div class='{fill_cls}' style='width:{prob*100:.1f}%;'></div>
              </div>
              <div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#484f58;'>
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            suggestions = {
                "high": [
                    "📞 Assign a dedicated retention agent immediately",
                    "🎁 Offer a loyalty discount or contract upgrade incentive",
                    "📋 Move to annual/biennial contract with price lock",
                    "🔒 Bundle security & tech support at no extra cost",
                ],
                "medium": [
                    "📧 Enroll in proactive check-in email campaign",
                    "💡 Highlight unused features the customer hasn't tried",
                    "🔔 Send satisfaction survey within 7 days",
                ],
                "low": [
                    "✅ Customer appears stable — maintain regular touchpoints",
                    "🌟 Eligible for referral rewards program",
                ],
            }
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-family:Space Grotesk,sans-serif;font-weight:600;color:#e6edf3;margin-bottom:0.8rem;'>
                Recommended Actions
              </div>
              {''.join(f"<div style='font-size:0.88rem;color:#8b949e;padding:0.3rem 0;'>{s}</div>" for s in suggestions[level])}
            </div>""", unsafe_allow_html=True)

        # SHAP for this prediction
        st.markdown("<div class='section-title'>Why This Prediction?</div>", unsafe_allow_html=True)
        try:
            import shap
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_input)[0]
            shap_df     = pd.DataFrame({
                "Feature": feature_names,
                "SHAP":    shap_vals,
                "Value":   X_input.iloc[0].values,
            }).reindex(pd.Index(range(len(feature_names))))
            shap_df["AbsSHAP"] = shap_df["SHAP"].abs()
            shap_df = shap_df.sort_values("AbsSHAP", ascending=False).head(12)

            colors  = ["#f85149" if v > 0 else "#3fb950" for v in shap_df["SHAP"]]
            fig = go.Figure(go.Bar(
                x=shap_df["SHAP"][::-1],
                y=shap_df["Feature"][::-1],
                orientation="h",
                marker_color=colors[::-1],
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e", family="Inter"),
                title=dict(text="SHAP Feature Contributions (red=↑churn, green=↓churn)",
                           font=dict(color="#e6edf3", size=13)),
                xaxis=dict(gridcolor="#30363d", title="SHAP Value"),
                yaxis=dict(gridcolor="#30363d"),
                height=400, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ex:
            st.info(f"SHAP explanation unavailable: {ex}")


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Predict
# ═════════════════════════════════════════════════════════════════
elif page == "📂 Batch Predict":
    st.markdown("<p class='hero-title' style='margin-bottom:0.2rem;'>📂 Batch <span style='color:#58a6ff;'>Customer Scoring</span></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.6rem;'>Upload a CSV with raw customer fields to score an entire cohort at once.</p>", unsafe_allow_html=True)

    st.markdown("<div class='info-box'>Your CSV should have the same columns as the original Telco dataset (excluding <code>Churn</code>). The app will add <code>churn_probability</code>, <code>churn_prediction</code>, and <code>risk_level</code> columns.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"<div class='section-title'>Preview — {len(df_raw):,} customers</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(5), use_container_width=True)

        with st.spinner("Scoring all customers…"):
            probs, preds, risks = [], [], []
            for _, row in df_raw.iterrows():
                raw = row.to_dict()
                raw.pop("Churn", None)
                X_row = preprocess_single(raw, scaler, feature_names)
                p = model.predict_proba(X_row)[0, 1]
                lv, lb, _, _ = risk_class(p)
                probs.append(round(p, 4))
                preds.append(int(p >= 0.5))
                risks.append(lb)

        df_out = df_raw.copy()
        df_out["churn_probability"] = probs
        df_out["churn_prediction"]  = preds
        df_out["risk_level"]        = risks

        st.markdown("<div class='section-title'>Scoring Summary</div>", unsafe_allow_html=True)
        total    = len(df_out)
        churners = sum(preds)
        high_r   = risks.count("🔴 HIGH RISK")
        med_r    = risks.count("🟡 MEDIUM RISK")
        low_r    = risks.count("🟢 LOW RISK")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Customers",  f"{total:,}")
        m2.metric("Predicted Churn",  f"{churners:,}")
        m3.metric("Churn Rate",       f"{churners/total*100:.1f}%")
        m4.metric("High Risk",        f"{high_r:,}")
        m5.metric("Medium Risk",      f"{med_r:,}")

        # Risk distribution chart
        risk_labels = ["🔴 HIGH RISK","🟡 MEDIUM RISK","🟢 LOW RISK"]
        risk_counts = [high_r, med_r, low_r]
        risk_colors = ["#f85149","#e3b341","#3fb950"]

        fig = go.Figure(go.Pie(
            labels=risk_labels, values=risk_counts,
            marker_colors=risk_colors,
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(color="#e6edf3", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            showlegend=False,
            title=dict(text="Risk Distribution", font=dict(color="#e6edf3", size=14)),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        col_chart, col_table = st.columns([1, 2])
        col_chart.plotly_chart(fig, use_container_width=True)

        col_table.markdown("<div class='section-title'>Results (top 20)</div>", unsafe_allow_html=True)
        show_cols = ["churn_probability","churn_prediction","risk_level"]
        if "customerID" in df_out.columns:
            show_cols = ["customerID"] + show_cols
        col_table.dataframe(df_out[show_cols].head(20), use_container_width=True)

        # Download
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Predictions CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ═════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("<p class='hero-title' style='margin-bottom:0.2rem;'>📊 Model <span style='color:#58a6ff;'>Performance</span></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.6rem;'>Evaluation metrics and diagnostic charts for the tuned XGBoost model on the held-out test set.</p>", unsafe_allow_html=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    ap  = average_precision_score(y_test, y_proba)

    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC",      f"{auc:.4f}")
    m2.metric("F1-Score",     f"{f1:.4f}")
    m3.metric("Avg Precision",f"{ap:.4f}")

    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Confusion Matrix", "Classification Report", "Feature Importance"])

    with tab1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={auc:.3f})",
                                  line=dict(color="#58a6ff", width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                  line=dict(color="#484f58", dash="dash")))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#8b949e"),
            title=dict(text="ROC Curve", font=dict(color="#e6edf3", size=14)),
            xaxis=dict(title="False Positive Rate", gridcolor="#30363d"),
            yaxis=dict(title="True Positive Rate",  gridcolor="#30363d"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            height=420, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        labels = ["No Churn","Churn"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#161b22"],[0.5,"#1f6feb"],[1,"#58a6ff"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False,
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#8b949e"),
            title=dict(text="Confusion Matrix", font=dict(color="#e6edf3", size=14)),
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual", autorange="reversed"),
            height=380, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        report = classification_report(y_test, y_pred,
                                        target_names=["No Churn","Churn"],
                                        output_dict=True)
        df_rep = pd.DataFrame(report).T.round(4)
        st.dataframe(df_rep.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                     use_container_width=True)

    with tab4:
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            imp = imp.sort_values(ascending=False).head(20)
            fig = go.Figure(go.Bar(
                x=imp.values[::-1], y=imp.index[::-1],
                orientation="h",
                marker_color="#58a6ff",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e"),
                title=dict(text="Top 20 Feature Importances", font=dict(color="#e6edf3", size=14)),
                xaxis=dict(gridcolor="#30363d", title="Importance"),
                yaxis=dict(gridcolor="#30363d"),
                height=520, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP Explainability
# ═════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainability":
    st.markdown("<p class='hero-title' style='margin-bottom:0.2rem;'>🔍 SHAP <span style='color:#58a6ff;'>Explainability</span></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.6rem;'>Global and local explanations showing which features drive churn predictions.</p>", unsafe_allow_html=True)

    try:
        import shap

        n_sample = st.slider("Samples to explain (more = slower)", 100, len(X_test), 300, 50)

        with st.spinner("Computing SHAP values…"):
            X_sample    = X_test.iloc[:n_sample].copy()
            X_sample.columns = feature_names
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_sample)

        tab1, tab2, tab3 = st.tabs(["Global Importance", "Beeswarm Summary", "Individual Customer"])

        with tab1:
            mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_names)
            mean_shap = mean_shap.sort_values(ascending=False).head(20)

            fig = go.Figure(go.Bar(
                x=mean_shap.values[::-1], y=mean_shap.index[::-1],
                orientation="h",
                marker=dict(
                    color=mean_shap.values[::-1],
                    colorscale=[[0,"#1f6feb"],[1,"#f85149"]],
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|", tickfont=dict(color="#8b949e")),
                ),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e"),
                title=dict(text="Global Feature Importance (Mean |SHAP|)", font=dict(color="#e6edf3", size=14)),
                xaxis=dict(title="Mean |SHAP Value|", gridcolor="#30363d"),
                yaxis=dict(gridcolor="#30363d"),
                height=540, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("<div class='info-box'>Beeswarm plot: each dot = one customer. Red = high feature value, blue = low. Dots to the right increase churn probability.</div>", unsafe_allow_html=True)
            fig_bsw, ax_bsw = plt.subplots(figsize=(10, 8))
            fig_bsw.patch.set_facecolor("#0d1117")
            ax_bsw.set_facecolor("#161b22")
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                               max_display=18, show=False)
            plt.gcf().patch.set_facecolor("#0d1117")
            for ax in plt.gcf().axes:
                ax.set_facecolor("#161b22")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e")
                ax.xaxis.label.set_color("#8b949e")
                ax.yaxis.label.set_color("#8b949e")
                ax.title.set_color("#e6edf3")
            plt.tight_layout()
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close()

        with tab3:
            idx = st.slider("Select customer index (from test set)", 0, n_sample - 1, 0)
            cust_prob = model.predict_proba(X_sample.iloc[[idx]])[0, 1]
            lv, lb, badge_cls, fill_cls = risk_class(cust_prob)
            st.markdown(f"""
            <div class='result-card' style='display:inline-block;padding:0.8rem 1.4rem;margin-bottom:1rem;'>
              <span style='color:#8b949e;font-size:0.86rem;'>Customer #{idx} · </span>
              <span class='risk-badge {badge_cls}' style='font-size:0.9rem;padding:0.3rem 0.9rem;'>{lb}</span>
              <span style='color:#58a6ff;font-family:Space Grotesk,sans-serif;font-weight:700;font-size:1.1rem;margin-left:0.8rem;'>{cust_prob*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

            cust_shap = shap_vals[idx]
            cust_df = pd.DataFrame({
                "Feature": feature_names,
                "Value":   X_sample.iloc[idx].values,
                "SHAP":    cust_shap,
            })
            cust_df["AbsSHAP"] = cust_df["SHAP"].abs()
            cust_df = cust_df.sort_values("AbsSHAP", ascending=False).head(15)

            colors = ["#f85149" if v > 0 else "#3fb950" for v in cust_df["SHAP"]]
            fig = go.Figure(go.Bar(
                x=cust_df["SHAP"][::-1], y=cust_df["Feature"][::-1],
                orientation="h",
                marker_color=colors[::-1],
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e"),
                title=dict(text=f"Customer #{idx} — Feature Contributions",
                           font=dict(color="#e6edf3", size=13)),
                xaxis=dict(title="SHAP Value  (+ = ↑churn risk)", gridcolor="#30363d",
                           zeroline=True, zerolinecolor="#484f58"),
                yaxis=dict(gridcolor="#30363d"),
                height=460, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.error("SHAP not installed. Run: `pip install shap`")
    except Exception as ex:
        st.error(f"SHAP error: {ex}")